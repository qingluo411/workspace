# 端到端性能**当前基线**（2026-10-10，#05 融合算子接入后）

> **用途：后续所有端到端改动都以本文数字为对照基准**（如无特别说明，直接与下表相减）。
> 同时记录 `dlc_kda_state_gather_scatter`（kernel `#8021` / pytorch `#2154`）接入后的现状，
> 作为 `下一步工作计划_2026-10-10.md` 的 M1–M8 在**无 profiler**条件下的补充基线。
>
> ⚠️ 与 `下一步工作计划` 的 M8（decode 602.6 ms/步）**不是同一测量条件**：那个是开
> profiler + `DLC_SYN_PROF_CYCLE=1` 的窗口，本文是无 profiler 的干净服务，两者不可直接相减。
> 服务：`vllm serve /mnt/jfs/models/GLM-5.3-Flash-Mix`，TP8、`--enforce-eager`、`--block-size 256`，
> **未开 profiler**（`DLC_SYN_PROF_CYCLE=0`，无 `--profiler-config`），`VLLM_LOGGING_LEVEL=INFO`。
> 进程启动时间 10:31:28 > `vllm_cl/patch/patch_kda.py` 修改时间 10:22:35，即本轮跑的是新 dispatch 代码
> （`DLC_KDA_STATE_POOL_OP=kernel` 默认生效）。

## 压测方式

`probes/bench_e2e_glm53.py`（流式，取 TTFT / e2e / 输出 token 速率），prompt 用
`/tmp/request_body_from_test_2048_3.json` 的正文（实际 **1223** token），每档重复取中位数，
并用 `/metrics` 的 `vllm:prompt_tokens_total` / `generation_tokens_total` 交叉核对。

**基线口径**（复现时保持一致，否则数字不可比）：
TP8 + `--enforce-eager` + `--block-size 256`、`VLLM_LOGGING_LEVEL=INFO`、**不开 profiler**、
`DLC_KDA_STATE_POOL_OP=kernel`、同一台机器同一批卡、脚本默认参数（`--repeat 2`）。

## 结果

**基线（2026-10-10，`DLC_KDA_STATE_POOL_OP=kernel`）**

| 用例 | 并发 | 墙钟 | TTFT | e2e | prompt tok | out tok | 输出 tok/s | prefill tok/s |
|---|---|---|---|---|---|---|---|---|
| 1223 tok prompt + 32 out | 1 | 11.55 s | 4724 ms | 11553 ms | 1223 | 32 | 2.95 | 112.6 |
| 1223 tok prompt + 64 out | 3 | 23.53 s | 5008 ms | 23526 ms | 1223 | 192 | 8.16 | 155.9 |
| 24 tok prompt + 128 out | 1 | 29.17 s | 364 ms | 29173 ms | 24 | 128 | **4.39** | 0.8 |

- 单请求 decode **≈233 ms/token（4.4 tok/s）**；3 并发 decode 聚合 **8.2 tok/s**。
- 1223 token prompt 的 TTFT ≈ 4.7 s（prefill ≈ 112–156 tok/s）。
- 说明：`下一步工作计划` 里的 602.6 ms/步（decode）是**开 profiler + cycle counter** 的窗口，
  与本次无 profiler 的 233 ms/token 不可直接相减比较。

## 关于 #05 的收益判定

- 调用点只在 **prefill 分支**（`vllm/models/glm5next/nvidia/kda.py:539` gather、`:563` scatter），
  decode 走 #04，所以 #05 不影响 decode 侧数字。
- 与接入前默认的「页表」Python 实现对比（离线实测，S=1、34 层）：
  页表 0.085 + 0.079 ms/层 → **34 层 5.57 ms**；融合算子 0.022 + 0.013 ms/层 → **34 层 0.76 ms**。
  即 prefill 步内**只省约 4.8 ms**，相对 ~4.7 s 的 TTFT 是 **0.1% 量级**，端到端不可分辨。
- 因此 #05 的价值不在 e2e 时间，而在：**少 68 次 kernel 下发/步**、**去掉 `torch.any/unique` 的 D2H**、
  以及**可被 `DLCGraph` 捕获**（为 D1 图捕获铺路）。

## 复现

```bash
cd /root/workspace/vllm-cl/docs/glm5_flash/probes
/root/miniconda/envs/qingluo-dev1/bin/python -u bench_e2e_glm53.py --tag kernel-8021 --repeat 2
```

如需与页表实现做 A/B，请以 `DLC_KDA_STATE_POOL_OP=fast` 重启服务后跑同一脚本；
若要与 `下一步工作计划` 的 M1–M8 对齐（下发次数 / 设备空洞 / 屏障空等），需以 `--profile`
（可选 `--kernel-log`）重启后取 trace。
