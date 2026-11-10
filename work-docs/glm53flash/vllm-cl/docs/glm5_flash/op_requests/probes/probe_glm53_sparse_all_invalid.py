"""#11 缺陷复现（2026-10-08）：512-NoPE 稀疏算子在「整行 topk 全为 -1」时输出 NaN。

现象：`torch.dlc_sparse_attention_kv_cache` 对 `topk_indices` 全 `-1` 的查询行返回
`out = NaN`（`lse = NaN`）。**同一输入重复调用会在「正确（全 0 / lse=-inf）」与「全 NaN」
之间交替**，且在 `kv_cache.size(0)` 分派出的六个 kernel 变体上都出现（与变体无关），
疑似全掩码行的规约读到了未初始化/未参与规约的暂存（首次调用往往正确、后续交替出错）。

对照（本脚本最后两行）：只要一行里有 **≥1 个有效槽**，同输入重复调用**逐位相同**
（实测 5 次 `bitwise_equal=True`），缺陷只落在「整行无效」的行上。

期望：整行无效 → 没有任何 key 可参与 softmax → 权重全 0 → `out = [Q,H,512]` **全 0**、
`lse = [Q,H]` **全 -inf**（且必须与调用次数无关）。判据来源有三个，本脚本都会跑：

  1. CPU golden reference：`vllm_cl/_custom_ops.py:sparse_attention_cpu`（softmax 后
     `nan_to_num(nan=0)` → 权重 0 → out 0）+ `sparse_attention_lse_from_gathered`
     （`logsumexp(全 -inf)` = `-inf`）；
  2. 插件 native 路径 `sparse_mla.py:_forward_native_nope`（同一约定，已用它对拍）；
  3. **同一算子家族的 576 分支** `custom_sparse_attention_kv_cache_bf16_*`：同输入
     3/3 次都是 `out` 全 0、`lse` 全 `-inf`、无 NaN —— 说明约定本来就该如此，
     只有 `_nope_` 变体没实现它。

为什么 `lse` 必须是 `-inf` 而不是 0：`lse` 由 DCP / `merge_attn_states` 消费，合并权重是
`exp(lse - max)`；`-inf` 才表示本段贡献为 0，写成 0 会被当成「权重 1」而污染结果。

影响：模型侧不触发。GLM-5.3-Flash 的索引器保证每个 token 至少 1 个有效槽
（首 token 取 incomplete tail 槽，`vllm_cl/patch/models/kpool_indexer.py:expand_pool_topk`），
2176 宽缓冲尾部的 125 列是 `-1` 而不是「整行 -1」。

用法：
    python probe_glm53_sparse_all_invalid.py dlc:0
    python probe_glm53_sparse_all_invalid.py dlc:0 --repeat 5     # 观察非确定性
判据：打出的 `out=0/lse=-inf` 为「正确」，出现 NaN 即复现缺陷。
"""
import argparse
import sys

import torch

LATENT, SCALE = 512, 512 ** -0.5


def run_one(dev, slots, valid, repeat, heads=8, q_tokens=8, topk=2176, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(q_tokens, heads, LATENT, generator=g, dtype=torch.bfloat16).to(dev)
    cache = torch.randn(slots, LATENT, generator=g, dtype=torch.bfloat16).to(dev)
    idx = torch.randint(0, slots, (q_tokens, topk), generator=g, dtype=torch.int32).to(dev)
    if valid == 0:
        idx.fill_(-1)
    else:
        idx[:, valid:] = -1
    results = []
    for _ in range(repeat):
        out, lse = torch.dlc_sparse_attention_kv_cache(q, cache, idx, SCALE, LATENT, True)
        torch.dlc.synchronize()
        out_f, lse_f = out.float(), lse.float()
        n_nan = int(torch.isnan(out_f).sum().item())
        lse_nan = int(torch.isnan(lse_f).sum().item())
        lse_neg_inf = int(torch.isinf(lse_f).sum().item())
        results.append(
            f"NaN(out={n_nan},lse={lse_nan})" if n_nan
            else (
                f"zeros(lse=-inf x{lse_neg_inf})"
                if int((out_f == 0).sum().item()) == out_f.numel()
                else "other"
            )
        )
    print(f"slots={slots:>8} valid={valid:>5} repeat={repeat} -> {results}", flush=True)
    return results


def cpu_golden():
    """CPU golden：整行 -1 时权重全 0 → out 全 0；lse = logsumexp(全 -inf) = -inf。"""
    import sys as _sys
    from pathlib import Path

    _sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from vllm_cl._custom_ops import (  # noqa: PLC0415
        sparse_attention_cpu,
        sparse_attention_lse_from_gathered,
    )

    g = torch.Generator().manual_seed(0)
    q = torch.randn(2, 8, LATENT, generator=g, dtype=torch.bfloat16)
    cache = torch.randn(64, LATENT, generator=g, dtype=torch.bfloat16)
    idx = torch.full((2, 2176), -1, dtype=torch.int32)
    out = sparse_attention_cpu(q, cache, idx, SCALE, LATENT)
    gathered = cache[idx.clamp(min=0).long()]
    lse = sparse_attention_lse_from_gathered(q, gathered, idx, SCALE, LATENT)
    return (f"out n_nan={int(torch.isnan(out.float()).sum())} "
            f"all_zero={bool((out == 0).all())} "
            f"lse all_-inf={bool(torch.isinf(lse.float()).all() and (lse.float() < 0).all())}")


def ref_576(dev, q_tokens=4, heads=8, topk=512, slots=4096):
    """同族 576 分支（latent 512 + rope 64）：应当 out 全 0、lse 全 -inf。"""
    g = torch.Generator().manual_seed(0)
    q = torch.randn(q_tokens, heads, LATENT + 64, generator=g,
                    dtype=torch.bfloat16).to(dev)
    cache = torch.randn(slots, LATENT + 64, generator=g,
                        dtype=torch.bfloat16).to(dev)
    idx = torch.full((q_tokens, topk), -1, dtype=torch.int32).to(dev)
    outs = []
    for _ in range(3):
        out, lse = torch.dlc_sparse_attention_kv_cache(
            q, cache, idx, SCALE, LATENT, True)
        torch.dlc.synchronize()
        out_f, lse_f = out.float(), lse.float()
        outs.append(
            f"n_nan={int(torch.isnan(out_f).sum())} "
            f"all_zero={bool((out_f == 0).all())} "
            f"lse_all_-inf={bool(torch.isinf(lse_f).all() and (lse_f < 0).all())}"
        )
    return f"h{heads}/topk{topk} x3 -> {outs}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dev", nargs="?", default="dlc:0")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    torch.dlc.set_device(int(args.dev.split(":")[-1]))
    print("=== 期望值参照（整行 -1）===", flush=True)
    print(f"  CPU golden (512-NoPE): {cpu_golden()}", flush=True)
    print(f"  576 分支同族算子     : {ref_576(args.dev)}", flush=True)
    print("=== 512-NoPE 变体实测 ===", flush=True)
    # 不同 kv_cache.size(0) 分派不同 kernel 变体（67584/204800/262144/1048576）。
    for slots in (4096, 67584, 70000, 204800, 262144, 300000):
        run_one(args.dev, slots, 0, args.repeat)
    # 对照：部分无效（尾部 125 列为 -1）必须稳定正确。
    run_one(args.dev, 262144, 2051, args.repeat)
    run_one(args.dev, 262144, 1, args.repeat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
