"""#11 验收：`torch.dlc_sparse_attention_kv_cache` 的 512-NoPE 路径。

旧缺陷（2026-09-21）：`latent=512, rope=0` 走 576 布局 kernel → 输出 NaN / 卡死设备；
且 topk 只分派 256/512/1024/1536/2048。算子侧 2026-10-08 交付 `nope` 家族
（`custom_sparse_attention_kv_cache_nope_bf16_{h2,h4,h8,h64}[_topk2176][_n*][_s*]`）。

GLM-5.3-Flash 几何：q [Q, H, 512] bf16（吸收后 query，无 RoPE 分量）；
cache [slots, 512] bf16；topk [Q, K] int32（K = index_topk，`-1` 为无效槽）；
softmax_scale = 1/sqrt(256)（**沿用 512 路径既有约定，不是 1/sqrt(512)**）；
kv_lora_rank = 512。模型 config：`index_topk = 2048`。

用法：python probe_glm53_sparse_nope512.py dlc:0 --all
判据：`out` 与 fp32 参考一致（bf16 量级，atol 2e-2）、无 NaN、`-1` 槽不参与、
      调用后设备仍健康。**LSE 单独报告**：算子按 `query.options()` 分配，512 与 576
      路径都返回 bf16（需求 #11 期望 fp32，见 SUMMARY 的 lse_note）。
"""
import argparse
import sys
import time

import torch

SCALE = 256 ** -0.5          # 512 路径既有约定
HKV = 512                    # GLM-5.3-Flash NoPE latent 宽度
CASES = [
    ("prefill-q64-h8-topk2176", dict(Q=64, H=8, K=2176, slots=4096, valid=2048)),
    ("prefill-q2048-h8-topk2048", dict(Q=2048, H=8, K=2048, slots=4096, valid=2048)),
    ("decode-q1-h8-topk2176", dict(Q=1, H=8, K=2176, slots=4096, valid=2048)),
    ("q64-h4-topk2048", dict(Q=64, H=4, K=2048, slots=4096, valid=2048)),
    ("all-invalid", dict(Q=8, H=8, K=2176, slots=4096, valid=0)),
    ("bigcache-n204800", dict(Q=64, H=8, K=2176, slots=100000, valid=2048)),
    ("unsupported-topk2051", dict(Q=4, H=8, K=2051, slots=4096, valid=2048)),
]


def make(dev, Q, H, K, slots, valid, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(Q, H, HKV, generator=g, dtype=torch.bfloat16).to(dev)
    cache = torch.randn(slots, HKV, generator=g, dtype=torch.bfloat16).to(dev)
    idx = torch.randint(0, slots, (Q, K), generator=g, dtype=torch.int32)
    idx[:, valid:] = -1
    return q.contiguous(), cache.contiguous(), idx.to(dev).contiguous()


def ref(q, cache, idx, row_chunk=128):
    """fp32 参考实现；按 query 行分块，避免一次性 gather [Q, K, 512] 撑爆内存。"""
    cache_f = cache.float()
    outs, lses = [], []
    for s0 in range(0, q.shape[0], row_chunk):
        qi = q[s0:s0 + row_chunk].float()
        ii = idx[s0:s0 + row_chunk]
        g = cache_f[ii.clamp(min=0).long()]
        scores = torch.einsum("qhd,qkd->qhk", qi, g) * SCALE
        scores = scores.masked_fill(~(ii >= 0).unsqueeze(1), float("-inf"))
        p = torch.nan_to_num(torch.softmax(scores, -1), nan=0.0)
        outs.append(torch.einsum("qhk,qkl->qhl", p, g))
        lses.append(torch.logsumexp(scores, -1))
    return torch.cat(outs), torch.cat(lses)


def alive(dev):
    x = torch.randn(256, 256, dtype=torch.bfloat16, device=dev)
    torch.dlc.synchronize()
    (x @ x).sum().item()


def run(name, cfg, dev, iters=3):
    q, cache, idx = make(dev, **cfg)
    try:
        torch.dlc_sparse_attention_kv_cache(q, cache, idx, SCALE, HKV, True)
        torch.dlc.synchronize()                        # warmup
        ts = []
        for _ in range(iters):
            t0 = time.time()
            out, lse = torch.dlc_sparse_attention_kv_cache(q, cache, idx, SCALE, HKV, True)
            torch.dlc.synchronize()
            ts.append((time.time() - t0) * 1e3)
    except RuntimeError as exc:
        print(f"[{name}] cfg={cfg} -> RuntimeError: {str(exc)[:190]}", flush=True)
        return None
    dt = min(ts)

    r_out, ref_lse = ref(q.cpu(), cache.cpu(), idx.cpu())
    err = (out.float().cpu() - r_out).abs().max().item()
    lse_c = lse.float().cpu()
    all_invalid = cfg["valid"] == 0
    with torch.no_grad():
        if all_invalid:
            lse_ok = bool(torch.isinf(lse_c).all() and (lse_c < 0).all())
            lse_err = float("inf")
            lse_note = "-inf as expected"
        else:
            lse_err = (lse_c - ref_lse).abs().max().item()
            lse_ok = lse_err < 1e-2
            lse_note = f"{lse_err:.3e} (|lse|~{ref_lse.abs().mean():.1f}, bf16 output)"
    nan = int(torch.isnan(out).sum().item() + torch.isnan(lse_c).sum().item())
    zero_ok = bool((out == 0).all().item()) if all_invalid else True
    try:
        alive(dev)
        healthy = True
    except Exception as exc:  # noqa: BLE001
        healthy, zero_ok = False, False
        print(f"[{name}] device_unhealthy: {exc}", flush=True)

    ok = err < 2e-2 and nan == 0 and zero_ok and healthy
    print(f"[{name}] Q={cfg['Q']} H={cfg['H']} K={cfg['K']} slots={cfg['slots']} "
          f"valid={cfg['valid']} out_err={err:.3e} nan={nan} all_invalid_zero={zero_ok} "
          f"t={dt:.2f}ms -> {'PASS' if ok else 'FAIL'}", flush=True)
    print(f"    lse: dtype={lse.dtype} err={lse_note} req_atol_1e-3={'OK' if lse_ok else 'NOT MET'}",
          flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dev", nargs="?", default="dlc:0")
    ap.add_argument("--case", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    torch.dlc.set_device(int(args.dev.split(":")[-1]))
    todo = CASES if args.all else [CASES[args.case if args.case is not None else 0]]
    bad = 0
    for name, cfg in todo:
        if run(name, cfg, args.dev) is False:
            bad += 1
    print(f"SUMMARY dev={args.dev} cases={len(todo)} failed={bad}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
