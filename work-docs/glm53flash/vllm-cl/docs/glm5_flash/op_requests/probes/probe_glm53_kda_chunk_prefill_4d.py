"""#03 验收：`torch.dlc_kda_chunk_prefill` 4-D ABI（GLM-5.3-Flash 真实几何）。

算子侧 2026-10-08 交付（kernel `dlc_kernels/attention/kda/dlc_kda_chunk_prefill.cpp`，
launch 名去掉 `2` 后缀），ABI 变为 **零 permute**：

    q/k/v/raw_g   [1, T, H, 128]  bf16
    beta_logits   [1, T, H]       bf16   （kernel 内做 sigmoid）
    a_log         [H]             fp32
    g_bias        [H, 128]        fp32   （不再 permute，旧签名是 [Dk, H]）
    initial_state [B, H, Dv, Dk]  fp32
    query_start_loc [B+1]         int32
    -> (out [1,T,H,Dv] bf16, final_state [B,H,Dv,Dk] fp32)

用法（一例一张干净卡，卡住就换卡并记为算子缺陷）：
    python probe_glm53_kda_chunk_prefill_4d.py dlc:0 --case 0
    python probe_glm53_kda_chunk_prefill_4d.py dlc:0 --all

判据：out/final_state 与 fp32 参考一致（bf16 量级，atol≈2e-2）；输入未被就地改写；
      调用后设备仍可跑普通算子。
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, "/root/workspace/vllm-cl/docs/glm5_flash/op_requests/refs")
from kda_reference_dv_dk import chunk_prefill_ref  # noqa: E402

H, DK, DV = 8, 128, 128
LB, SCALE = -5.0, DK ** -0.5
CHUNK = 64

# (name, lens, heads, chunk) —— heads 覆盖 TP8/16/32 的本地头数（8/4/2）
CASES = [
    ("glm53-prefill-2048", [2048], 8, CHUNK),
    ("glm53-prefill-ragged-2x1024", [1024, 1024], 8, CHUNK),
    ("ragged-3seq", [512, 100, 1], 8, CHUNK),
    ("single-chunk", [64], 8, CHUNK),
    ("non-multiple-of-chunk", [257], 8, CHUNK),
    ("short-1token", [1], 8, CHUNK),
    ("heads-4", [512], 4, CHUNK),
    ("heads-2", [512], 2, CHUNK),
    ("chunk-16", [512], 8, 16),
    ("chunk-32", [512], 8, 32),
    ("chunk-128", [512], 8, 128),
]


def make(lens, dev, heads=H, seed=0):
    g = torch.Generator().manual_seed(seed)
    n, b = sum(lens), len(lens)
    r = lambda *s: torch.randn(*s, generator=g)
    to = lambda t, dt: t.to(dtype=dt, device=dev).contiguous()
    return dict(
        q=to(r(1, n, heads, DK) * 0.5, torch.bfloat16),
        k=to(r(1, n, heads, DK) * 0.5, torch.bfloat16),
        v=to(r(1, n, heads, DV) * 0.5, torch.bfloat16),
        raw_g=to(r(1, n, heads, DK) * 0.5, torch.bfloat16),
        beta_logits=to(r(1, n, heads) * 0.5, torch.bfloat16),
        a_log=to(r(heads) * 0.5, torch.float32),
        g_bias=to(r(heads, DK) * 0.2, torch.float32),
        initial_state=to(r(b, heads, DV, DK) * 0.05, torch.float32),
        query_start_loc=torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(lens, dtype=torch.int64), 0)),
            dtype=torch.int32, device=dev),
    )


def alive(dev):
    """调用后设备仍能跑普通算子（返回 None 表示健康，抛异常/挂住即异常）。"""
    x = torch.randn(256, 256, dtype=torch.bfloat16, device=dev)
    torch.dlc.synchronize()
    (x @ x).sum().item()


def run(name, lens, heads, chunk, dev, iters=3):
    a = make(lens, dev, heads)
    snap = {k: v.clone() for k, v in a.items()}
    call = lambda: torch.dlc_kda_chunk_prefill(  # noqa: E731
        a["q"], a["k"], a["v"], a["raw_g"], a["beta_logits"], a["a_log"],
        a["g_bias"], a["initial_state"], a["query_start_loc"],
        float(LB), float(SCALE), int(chunk))
    call()
    torch.dlc.synchronize()                                   # warmup
    ts = []
    for _ in range(iters):
        t0 = time.time()
        out, fin = call()
        torch.dlc.synchronize()
        ts.append((time.time() - t0) * 1e3)
    dt = min(ts)

    a_cpu = {k: v.cpu() for k, v in a.items()}
    ref_out, ref_state = chunk_prefill_ref(
        a_cpu["q"], a_cpu["k"], a_cpu["v"], a_cpu["raw_g"], a_cpu["beta_logits"],
        a_cpu["a_log"], a_cpu["g_bias"], a_cpu["initial_state"], a_cpu["query_start_loc"],
        lower_bound=LB, query_scale=SCALE, chunk_size=chunk)
    o_err = (out.float().cpu() - ref_out).abs().max().item()
    s_err = (fin.cpu() - ref_state).abs().max().item()
    dirty = [k for k in snap if not torch.equal(snap[k].cpu(), a[k].cpu())]

    try:
        alive(dev)
        healthy = True
    except Exception as exc:  # noqa: BLE001
        healthy, dirty = False, dirty + [f"device_unhealthy: {exc}"]
    ok = o_err < 2e-2 and s_err < 2e-2 and not dirty and healthy
    print(f"[{name}] lens={lens} H={heads} chunk={chunk} out_err={o_err:.3e} state_err={s_err:.3e} "
          f"inplace_modified={dirty or 'none'} t={dt:.2f}ms -> {'PASS' if ok else 'FAIL'}",
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
    for name, lens, heads, chunk in todo:
        if not run(name, lens, heads, chunk, args.dev):
            bad += 1
    print(f"SUMMARY dev={args.dev} cases={len(todo)} failed={bad}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
