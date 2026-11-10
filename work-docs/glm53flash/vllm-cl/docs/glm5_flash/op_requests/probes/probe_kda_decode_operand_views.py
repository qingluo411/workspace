"""逐个操作数：模型侧的「view 形态」能不能直接喂 `dlc_kda_recurrent_decode`？

回答的问题：`_fused_operand`（`vllm_cl/patch/patch_kda.py`）为什么对 9 个操作数里
的某几个必须物化、对另几个只是「省一次隐式 staging」。

三段实验（一次跑完）：

  1. **入口校验**：把单个操作数换成模型侧的真实产生方式（其余用规范张量），看是
     硬报错还是能过。硬报错的就**必须物化**，否则整条链路回落 native。
  2. **隐式 staging 成本**：`query_start_loc` / `state_indices` 用「常驻 buffer 的切片」
     传时能过校验，但 launcher 会 `to_contiguous_dlc` 隐式拷一份；对比「新鲜连续张量」
     的单次耗时差。
  3. **kernel 边界**：`B == 128` 时 kernel 的 metadata cache 只装 128 个 int32 却要读
     第 `B` 个（第 129 个）→ Assert。顺带在脚本里验一下。

结论（2026-10-09，dlc:0）：
  * q/k/v = `qkv_ns.split(HD)[i].reshape(1,T,H,D)` → `:72` 不连续（T>1）/ `:76` offset（T=1 时 k/v）
  * raw_g = `f_b_proj(f_a)[0].reshape(1,-1,H,D)` → `:104-107` 512B 物理行
  * beta  = 合并 GEMM 输出切片 `+unsqueeze(0)` → `:72`（切片不连续）或 `:76`（offset）
  * A_log / g_bias = 根参数 reshape → 直接可用，无需物化
  * query_start_loc / state_indices → 能过校验且结果正确；但视图形态每次调用多约
    10~19 µs（launcher 隐式 staging），B 不是 128 的整数倍时都存在
  * `B == 128` 会让 kernel 断言失败（与本脚本第 3 段）

用法：
    cd /root/workspace/vllm-cl
    python -u docs/glm5_flash/op_requests/probes/probe_kda_decode_operand_views.py dlc:0
"""
import sys
import time

import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else "dlc:0"
torch.dlc.set_device(int(DEV.split(":")[-1]))

H, D = 8, 128


def bf16(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.5).to(
        torch.bfloat16
    ).to(DEV)


def fp32(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g, dtype=torch.float32) * 0.25).to(
        torch.float32
    ).to(DEV)


def make(B):
    """规范形态的一套输入 + in-place 池 + out。"""
    return dict(
        q=bf16(1, B, H, D, seed=1), k=bf16(1, B, H, D, seed=2), v=bf16(1, B, H, D, seed=3),
        raw_g=bf16(1, B, H, D, seed=4), beta=bf16(1, B, H, seed=5),
        a_log=fp32(H, seed=6), g_bias=fp32(H, D, seed=7),
        state=fp32(B + 1, H, D, D, seed=9),
        idx=torch.arange(1, B + 1, dtype=torch.int32).to(DEV),
        qsl=torch.arange(B + 1, dtype=torch.int32).to(DEV),
    )


def call(a, **over):
    t = dict(a, **over)
    return torch.dlc_kda_recurrent_decode(
        t["q"], t["k"], t["v"], t["raw_g"], t["beta"], t["a_log"], t["g_bias"],
        t["state"], t["idx"], t["qsl"], None,
        lower_bound=-5.0, scale=D ** -0.5, out=t.get("out", torch.empty_like(t["v"])),
    )


def probe(name, a, **over):
    try:
        call(a, **over)
        torch.dlc.synchronize()
        print(f"  {name:52s} OK（能过入口校验）")
    except RuntimeError as e:
        first = next(
            (ln for ln in str(e).strip().splitlines() if ".cpp:" in ln), str(e)
        )
        loc = first.split("(")[-2].split("/")[-1]
        msg = first.split(") ", 1)[-1][:60]
        print(f"  {name:52s} FAIL [{loc}] {msg}")


print(f"=== 1. 入口校验（设备 {DEV}，H={H} D={D}）===")
B = 2
a = make(B)
probe("control：全部独立 [1,T,H,D] 张量", a)
qkv = bf16(1, B, 3 * H * D, seed=11)
q3, k3, v3 = (x.reshape(1, B, H, D) for x in qkv.split(H * D, dim=-1))
probe("q/k/v = split(3HD)[i].reshape(1,T,H,D)", a, q=q3, k=k3, v=v3)
probe("raw_g = reshape of [T, H*D]（f_b_proj 输出）", a, raw_g=bf16(B, H * D, seed=13).reshape(1, B, H, D))
probe("raw_g = reshape of [1, T, H*D]", a, raw_g=bf16(1, B, H * D, seed=14).reshape(1, B, H, D))
merged = bf16(B, 3 * H, seed=21)
probe("beta = merged[T,3H].split(H)[1].unsqueeze(0)", a, beta=merged.split(H, dim=-1)[1].unsqueeze(0))
merged1d = bf16(B * 3 * H, seed=22)
probe("beta = merged[1,T*3H].split[T*H..].reshape(1,T,H)", a, beta=merged1d.split(B * H)[1].reshape(1, B, H))
big = torch.arange(129, dtype=torch.int32).to(DEV)
bt = torch.arange(128, dtype=torch.int32).to(DEV).reshape(128, 1) + 1
probe("query_start_loc = big[:B+1]（常驻 [129] 前缀切片）", a, qsl=big[: B + 1])
probe("state_indices  = bt[:B, 0]（1-D 列视图）", a, idx=bt[:B, 0])
probe("A_log / g_bias = 根参数 reshape", a,
      a_log=fp32(1, 1, H, 1, seed=6).reshape(-1), g_bias=fp32(H * D, seed=7).reshape(H, D))

print(f"\n=== 2. 视图 vs 新鲜连续张量：单次耗时（B=64）===")
B = 64
a = make(B)
bt = torch.arange(128, dtype=torch.int32).to(DEV).reshape(128, 1) + 1
big = torch.arange(129, dtype=torch.int32).to(DEV)
a["out"] = torch.empty_like(a["v"])
cases = {
    "qsl=clone  idx=clone": (big[: B + 1].clone(), a["idx"].clone()),
    "qsl=view   idx=clone": (big[: B + 1], a["idx"].clone()),
    "qsl=clone  idx=view 1-D": (big[: B + 1].clone(), bt[:B, 0]),
    "qsl=clone  idx=view [B,1]": (big[: B + 1].clone(), bt[:B]),
}
best = {k: 1e9 for k in cases}
IT, ROUNDS = 200, 8


def run(ql, ix, n):
    for _ in range(n):
        call(a, qsl=ql, idx=ix)


for _ in range(ROUNDS):
    for name, (ql, ix) in cases.items():
        run(ql, ix, 20)
        torch.dlc.synchronize()
        t0 = time.perf_counter()
        run(ql, ix, IT)
        torch.dlc.synchronize()
        best[name] = min(best[name], (time.perf_counter() - t0) / IT * 1e6)
base_us = best["qsl=clone  idx=clone"]
for name, us in best.items():
    print(f"  {name:28s} {us:8.1f} us/call   Δ={us - base_us:+6.1f}")

print("\n=== 3. kernel 边界：B 与 metadata cache（kVectorWords=128）===")
for B in (126, 127, 128, 129):
    a = make(B)
    try:
        call(a)
        torch.dlc.synchronize()
        print(f"  B={B:4d} OK")
    except Exception as e:
        print(f"  B={B:4d} FAIL {str(e).splitlines()[0][:60]}")
