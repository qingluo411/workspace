"""dlc_mhc_pre / dlc_mhc_post vs vLLM native at GLM-5.3-Flash real geometry.

Geometry taken from the checkpoint config and from layer-0 weights:
  hidden = 4096, hc_mult = 4, hc_mult3 = 24
  mhc_sinkhorn_iterations = 20, hc_eps = 1e-6, rms_norm_eps = 1e-5
  mhc_post_mult_value = 2.0

Reference: vllm/model_executor/kernels/mhc/torch.py (the exact native path the
DLC plugin falls back to today).

Findings (2026-09-28, dlc:0):

  Correctness at repeat=20 / hidden=4096 / hc_mult=4 / layer-0 real weights:
    post_mix     err ~1.3e-04 (|ref|~1.9)   rounding level
    comb_mix     err ~6.1e-05 (|ref|~1.0)   rounding level
    layer_input  err  7.812e-03 = 1 bf16 ULP at |ref|~1.8
    mhc_post     err  0.000e+00 (bit-exact)
    dlc_mhc_post rejects a [N,4,4] comb (must be flat [N,16])
  => numerically safe to wire in; the 20-iteration Sinkhorn loop matches native.

  Performance: dlc_mhc_post wins at every N. dlc_mhc_pre is LINEAR in N at
  ~0.19 ms/token and that cost is inside the kernel (the fp32 cast is 0.58 ms
  and the equivalent torch.matmul 1.5 ms at N=4096), so it only pays off for
  small N:

    N        1      2      4      8     16     32     64   4096
    pre     8.2x    -      -    1.99x  0.99x  0.59x  0.34x  0.04x
    post    2.1x    -      -    7.2x    -      -    30.0x  48.2x
    chain   6.3x   6.8x   3.7x  2.0x   0.99x  0.59x  0.44x   -
    (chain = post + pre, i.e. MHCFusedPostPreOp.forward_native; x = native/dlc)

  => the plugin must gate dlc_mhc_pre on N (small decode batch) and keep the
     native path for prefill. The ~0.19 ms/token (~4 GFLOPS) inside the kernel
     is a performance defect worth reporting: torch.matmul does the same GEMM
     ~500x faster at N=4096.
"""

import sys
import time

import torch
from safetensors import safe_open
from vllm.model_executor.kernels.mhc import mhc_post_torch, mhc_pre_torch

DEV = sys.argv[1] if len(sys.argv) > 1 else "dlc:0"
MODEL = "/mnt/jfs/models/GLM-5.3-Flash-Mix"
SHARD = f"{MODEL}/model-00001-of-00120.safetensors"

HIDDEN = 4096
HC = 4
HC2 = HC * HC
HC3 = 2 * HC + HC2
SINKHORN = 20
HC_EPS = 1e-6
RMS_EPS = 1e-5
POST_MULT = 2.0


def load_layer0_params():
    out = {}
    with safe_open(SHARD, framework="pt") as f:
        for name in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale"):
            key = f"model.language_model.layers.0.{name}"
            out[name] = f.get_tensor(key).to(torch.float32)
    return out["hc_attn_fn"], out["hc_attn_base"], out["hc_attn_scale"]


def sync():
    try:
        torch.dlc.synchronize()
    except Exception:
        torch.cuda.synchronize()


def maxerr(a, b):
    a = a.float()
    b = b.float()
    return (a - b).abs().max().item(), b.abs().max().item()


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    torch.manual_seed(0)
    fn_w, base, scale = load_layer0_params()
    print(
        f"# device={DEV} fn={tuple(fn_w.shape)} base={tuple(base.shape)} "
        f"scale={tuple(scale.shape)} sinkhorn={SINKHORN} post_mult={POST_MULT}"
    )
    print(f"# layer0 scale={scale.tolist()} base_absmax={base.abs().max().item():.4f}")

    for N in (1, 64, 512, 4096):
        residual = torch.randn(N, HC, HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.5
        fn = fn_w.to(DEV).contiguous()
        hc_scale = scale.to(DEV).contiguous()
        hc_base = base.to(DEV).contiguous()
        x = torch.randn(N, HIDDEN, dtype=torch.bfloat16, device=DEV) * 0.5

        # ---------- pre ----------
        p_dlc, c_dlc, li_dlc = torch.dlc_mhc_pre(
            residual, fn, hc_scale, hc_base,
            RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN, 1,
        )
        p_ref, c_ref, li_ref = mhc_pre_torch(
            residual, fn, hc_scale, hc_base,
            RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
        )
        sync()
        p_err = maxerr(p_dlc, p_ref.reshape(N, HC))
        c_err = maxerr(c_dlc, c_ref.reshape(N, HC2))
        li_err = maxerr(li_dlc, li_ref)
        li_bit = bool((li_dlc.view(torch.int16) == li_ref.view(torch.int16)).all())

        # ---------- post ----------
        post = p_ref.reshape(N, HC).contiguous()
        comb = c_ref.reshape(N, HC2).contiguous()
        o_dlc = torch.dlc_mhc_post(x, residual, post, comb)
        o_ref = mhc_post_torch(x, residual, post.reshape(N, HC, 1), comb.reshape(N, HC, HC))
        sync()
        o_err = maxerr(o_dlc, o_ref)

        # post must reject the un-flattened [N,HC,HC] comb
        rejected = "no"
        try:
            torch.dlc_mhc_post(x, residual, post, comb.reshape(N, HC, HC))
            rejected = "NO(!)"
        except Exception:
            rejected = "yes"

        # ---------- fused chain: dlc post -> dlc pre ----------
        r2 = torch.dlc_mhc_post(x, residual, post, comb)
        p2, c2, li2 = torch.dlc_mhc_pre(
            r2, fn, hc_scale, hc_base,
            RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN, 1,
        )
        r2n = mhc_post_torch(x, residual, post.reshape(N, HC, 1), comb.reshape(N, HC, HC))
        p2n, c2n, li2n = mhc_pre_torch(
            r2n, fn, hc_scale, hc_base,
            RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN,
        )
        sync()
        ch_err = maxerr(li2, li2n)

        print(
            f"N={N:5d} pre: post_err={p_err[0]:.3e}/{p_err[1]:.2f} "
            f"cm_err={c_err[0]:.3e}/{c_err[1]:.2f} "
            f"li_err={li_err[0]:.3e}/{li_err[1]:.2f} li_bit_exact={li_bit} | "
            f"post: out_err={o_err[0]:.3e}/{o_err[1]:.2f} reject_3d_comb={rejected} | "
            f"chain li_err={ch_err[0]:.3e}/{ch_err[1]:.2f}"
        )

        if N in (1, 4096):
            t_pre_dlc = bench(lambda: torch.dlc_mhc_pre(
                residual, fn, hc_scale, hc_base,
                RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN, 1))
            t_pre_nat = bench(lambda: mhc_pre_torch(
                residual, fn, hc_scale, hc_base,
                RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN))
            t_post_dlc = bench(lambda: torch.dlc_mhc_post(x, residual, post, comb))
            t_post_nat = bench(lambda: mhc_post_torch(
                x, residual, post.reshape(N, HC, 1), comb.reshape(N, HC, HC)))
            print(
                f"    N={N:5d} pre  dlc={t_pre_dlc:.3f}ms native={t_pre_nat:.3f}ms "
                f"speedup={t_pre_nat / t_pre_dlc:.1f}x | "
                f"post dlc={t_post_dlc:.3f}ms native={t_post_nat:.3f}ms "
                f"speedup={t_post_nat / t_post_dlc:.1f}x"
            )


if __name__ == "__main__":
    main()
