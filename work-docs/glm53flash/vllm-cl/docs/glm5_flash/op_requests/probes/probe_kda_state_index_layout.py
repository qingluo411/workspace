"""`state_indices` 布局实验：`block_table[:, 0]` 能不能不拷贝直接喂 KDA kernel？

运行时形态（`mamba_cache_mode="none"`，无 spec）：
  * `BlockTable.block_table.gpu` = `[max_num_reqs, 1]` int32（`max_num_blocks_per_req`
    在 none 模式下等于 `1 + num_speculative_blocks`，见 `kv_cache_interface.py`）；
  * `non_spec_state_indices_tensor = block_table_tensor[:, 0]`（`gdn_attn.py:257`）。

DLC 存储描述符对 `[128, 1]` int32 是 `dim0=1 / dim0_padded=128`（int32 alignment=128
元素=512B），即**每个请求的索引独占一条 512B 物理行、只用了前 4B**。因此 `[:, 0]`
这个 `stride=(1,)` 的视图与物理布局不一致：`is_contiguous_dlc` 为假，launcher 会
`to_contiguous_dlc` 隐式拷一份（与 `_fused_operand` 的 clone 等价）。

本脚本在真机上比较三种传法（#04 recurrent decode / #02 causal_conv_update 各一组）：
  A) `bt[:B, 0].clone()`  —— 现状（`_fused_operand` / `_update_new_op` 的路径）
  B) `bt[:B]`             —— 2-D `[B, 1]` 切片（**2026-10-09 修正**：`is_contiguous_dlc` 只在
                              `B % 128 == 0` 时为真；一般 batch 下它与 C 一样会被 launcher
                              `to_contiguous_dlc` 隐式拷一份，实测 +19.1 µs/call@B=64，
                              见 `probe_kda_decode_operand_views.py`）
  C) `bt[:B, 0]`          —— 原始列视图（不拷），看 launcher 是否自己兜住

判定口径：A 已由 `tests/test_glm5_kda_recurrent_decode_op.py` / `..._conv_update_op.py`
验证过与 fp32 reference 一致，所以 B/C 只要与 A **逐位一致**即视为正确实现可直接用。

用法：
    python -u probe_kda_state_index_layout.py dlc:0
"""
import sys
import time

import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else "dlc:0"
torch.manual_seed(0)

MAX_REQS = 128          # vLLM 默认 max_num_seqs=128
B = 4                   # 本步 decode 请求数
H, DK, DV = 8, 128, 128
LB, SCALE = -5.0, 128 ** -0.5


def block_table(slots):
    """模拟常驻 block table：`[128, 1]` int32，前 B 行是真实槽位。"""
    bt = torch.zeros(MAX_REQS, 1, dtype=torch.int32, device=DEV)
    bt[:B, 0] = torch.tensor(slots, dtype=torch.int32, device=DEV)
    return bt


def variants(bt):
    return {
        "A clone(bt[:B,0])": bt[:B, 0].clone(),
        "B bt[:B] ([B,1])": bt[:B],
        "C bt[:B,0] (view)": bt[:B, 0],
        "D dense fresh [B]": bt[:B, 0].clone().clone(),
    }


def _rand(shape, scale):
    return (torch.randn(*shape, dtype=torch.float32, device=DEV) * scale).to(torch.bfloat16)


def build_recurrent():
    q = _rand((1, B, H, DK), 0.5)
    k = _rand((1, B, H, DK), 0.5)
    v = _rand((1, B, H, DV), 0.5)
    raw_g = _rand((1, B, H, DK), 0.25)
    beta = _rand((1, B, H), 0.5)
    a_log = (torch.randn(H, dtype=torch.float32, device=DEV) * 0.25).contiguous()
    g_bias = (torch.randn(H, DK, dtype=torch.float32, device=DEV) * 0.25).contiguous()
    state = torch.randn(B + 1, H, DV, DK, dtype=torch.float32, device=DEV) * 0.05
    qsl = torch.arange(B + 1, dtype=torch.int32, device=DEV)
    return dict(q=q, k=k, v=v, raw_g=raw_g, beta=beta, a_log=a_log,
                g_bias=g_bias, state=state, qsl=qsl)


def run_recurrent(inputs, sidx):
    out = torch.empty(1, B, H, DV, dtype=torch.bfloat16, device=DEV)
    state = inputs["state"].clone()
    torch.dlc_kda_recurrent_decode(
        inputs["q"], inputs["k"], inputs["v"], inputs["raw_g"], inputs["beta"],
        inputs["a_log"], inputs["g_bias"], state, sidx, inputs["qsl"],
        None, lower_bound=LB, scale=SCALE, out=out)
    torch.dlc.synchronize()
    return out.clone(), state.clone()


def build_conv():
    D = 3 * 1024
    x = _rand((B, 1, D), 0.5).contiguous()
    w4d = (torch.randn(D, 4, dtype=torch.float32, device=DEV) * 0.3).t().contiguous()
    state = torch.randn(B + 1, 3, D, dtype=torch.bfloat16, device=DEV) * 0.5
    return dict(x=x, w4d=w4d, state=state)


def run_conv(inputs, sidx):
    state = inputs["state"].clone()
    out = torch.dlc_kda_causal_conv_update(
        inputs["x"], inputs["w4d"], state, sidx,
        max_query_len=1, activation="silu", null_state_id=0)
    torch.dlc.synchronize()
    return out.clone(), state.clone()


def compare(tag, run, inputs, bt):
    base = None
    for name, sidx in variants(bt).items():
        try:
            for _ in range(3):          # warmup（首调用含 kernel 首次下发）
                run(inputs, sidx)
            samples = []
            for _ in range(20):
                t0 = time.perf_counter()
                out, st = run(inputs, sidx)
                samples.append((time.perf_counter() - t0) * 1e3)
            samples.sort()
            dt = samples[len(samples) // 2]
            spread = f"{samples[0]:.3f}~{samples[-1]:.3f}"
            if base is None:
                base, ref = (out, st), name
                print(f"[{tag}] {name:22s} ok  med={dt:7.3f}ms [{spread}]  (reference)")
            else:
                oeq = torch.equal(out.view(torch.int16), base[0].view(torch.int16))
                seq = torch.equal(st.view(torch.int16), base[1].view(torch.int16))
                print(f"[{tag}] {name:22s} ok  med={dt:7.3f}ms [{spread}]  "
                      f"out=={ref}: {oeq}  state=={ref}: {seq}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{tag}] {name:22s} ERR {type(exc).__name__}: {str(exc)[:160]}")


if not hasattr(torch, "dlc_kda_recurrent_decode"):
    print("no dlc_kda ops on this build"); raise SystemExit(1)

print(f"dev={DEV} B={B} max_num_reqs={MAX_REQS} H={H} Dk={DK} Dv={DV}")
print("=== #04 dlc_kda_recurrent_decode ===")
compare("decode", run_recurrent, build_recurrent(), block_table([1, 2, 3, 4]))
print("=== #02 dlc_kda_causal_conv_update ===")
compare("conv", run_conv, build_conv(), block_table([1, 2, 3, 4]))
