# 覆盖核查实测脚本（probes）

本目录是 2026-09-21「已有算子覆盖核查」用的实测脚本，结论汇总见
[`../00_已有算子覆盖核查.md`](../00_已有算子覆盖核查.md)。
由这些实测写成的、可直接贴成 Issue 的修复单：
[`../给算子组_01_02_修复要点.md`](../给算子组_01_02_修复要点.md)、
[`../给算子组_03_04_修复要点.md`](../给算子组_03_04_修复要点.md)、
[`../给算子组_05_state_pool_index_perf.md`](../给算子组_05_state_pool_index_perf.md)（2026-09-28，真机 profile）。

## 运行方式

```bash
# 前提：DLC 设备上没有正在运行的模型服务（否则显存/HugePages 会互相干扰）
cd /root/workspace/vllm-cl
/root/miniconda/envs/qingluo-dev1/bin/python -u docs/glm5_flash/op_requests/probes/probe_mhc_pre_post.py
```

- 输出里的 `[DLCSynapse] ... HugePages_Free=0` 行属于设备侧噪声，可忽略；
- **一次只跑一个脚本**：多个进程同时占用 `dlc:0` 时可能出现互相拖慢甚至卡死；
- 每个脚本自带 reference 实现并打印 `*_err`，`err` 在 bf16 量级（~1e-3）即视为数值一致。

## 脚本与记录到的结果

| 脚本 | 对应条目 | 记录结果（2026-09-21） |
|---|---|---|
| `probe_op_surface.py` | 全部 | 打印存在/缺失算子清单；`silu_and_mul_with_clamp`、`dlc_mhc_fused_post_pre`、`dlc_kpool_compress` 等为 MISSING |
| `probe_concat_and_cache_mla_nope.py` | #12 | `kpe=64` 与 `kpe=0` 均 `latent_err=0.000e+00 kpe_err=0.000e+00` |
| `probe_sparse_attention_nope512.py <latent> <rope> <topk>` | #11 | `512/64/2048` OK err 3.91e-03；`512/0/2048` 输出 NaN（复测会卡死）；`512/0/2051`、`512/0/2176` 报 `only supports topk_tokens 256/512/1024/1536/2048` |
| `probe_causal_conv1d_prefill_update.py` | #01 / #02 | prefill `out_err=1.953e-03 state_err=0.000e+00`；其中 update 部分是 2026-09-21 初版写法，其 page 0 结论已被修正版取代 |
| `probe_causal_conv1d_state_index_zero.py`（2026-09-22 修正版） | #02 | 真实页 `[1]/[2]/[3]/[-1,1]` 均 `out_err=0.00e+00 state_err=0.00e+00`；`[0]` 与 `[-1]` 为 skipped（`out_zero=True`，page 0 另 `state_unchanged=True`） |
| `probe_kda_chunk_gate_convention.py` | #03 | `exp(g)` 语义吻合（out_err 1.39e-03），`exp2(g)` 不吻合 |
| `probe_kda_chunk_reference_compare.py` | #03 | 逐 token g 与 fp32 递推 err 1.02e-04；cumsum 输入则不吻合 |
| `probe_kda_chunk_perchannel_rejected.py` | #03 | per-head gate OK；逐通道 `[1,T,H,D]` 报 `got dim 4`，`[T,H,K]` 报 `batch size must be 1` |
| `probe_kda_recurrent.py` | #04 | per-head `out_err=7.77e-04 state_err=1.88e-03`；逐通道 gate 被参数校验拒绝 |
| `probe_fused_rms_norm_packed_rows.py` | #17 | `dlc_fused_rms_norm_f32_bf16` 对 2D packed 行归一化 err 6.02e-04（fp32 入 / bf16 weight / fp32 出） |
| `probe_index_select_scatter.py` | #05 | ~~`index_select+where` 与 `index_copy_` 在 4D state 上均 OK~~ **结论作废（2026-09-28）**：该脚本用的是 `[8,4,16,16]`（8 KiB）小张量，没覆盖真实 `[1168,8,128,128] fp32` 页内非连续 view。改用下方脚本 |
| `probe_state_pool_index_cost.py [dlc:N]` | #05 | **2026-09-28 新增（P0 复现）**：同一批 index／同样只动 8 页，池子换成真实 Mamba 页内 view（`dim0 stride=135680 ≠ 131072`）后 —— `index_select+clone` 0.181 ms → **579.7 ms（3196x）**、`index_copy_` 1.544 ms → **1873.0 ms（1213x）**；34 层累计 58.7 ms → **83.4 s/步**。退出码 0 = 复现。另附 dense 池的页数扫描（4→584 MiB：0.083→1.544 ms）证明 `index_copy_` 整池 DMA |
| `probe_mhc_pre_post.py` | #13 / #14 | pre：post 3.03e-05 / cm 1.06e-04 / layer_input 3.92e-03；post：3.90e-03 |
| `probe_mqa_logits_semantics.py` | #09 | 与 KPool 语义误差 23.58，与「先按 head 加权再点积」误差 0.0 |
| `probe_zero_block_ids.py` | #18 | 官方 int32 参数形态 `exact_match=True`；bf16 页（page_el 按 int32 计 64 = 256B）`exact_match=False`——**该用例是人为构造的「非 512B 对齐页」反例，GLM5 真实页不会出现** |
| `probe_zero_block_ids_page_alignment.py` | #18 | 页大小非 128 字（512B）整数倍时，实际被清零的块是 1,3,5,7（期望 1,3）——`ALIGN128` 放大了 block 步长（同为反例；约束本身与 GLM5 不冲突） |
| `probe_zero_block_ids_glm5_geometry.py` | #18 | 2026-09-22 新增：按 GLM5 真实 5 类页（4,096 / 18,432 / 32,768 / 262,144 / 524,288 B）各建 3 个 segment 实清 → 全部 `exact_match=True`（`ALL_ALIGNED_AND_EXACT: True`）；另测出「同一次调用混用不同页大小会清错相邻分配」 |
| `probe_new_kda_conv_prefill.py <dev> [case_ids]` | #01 | （**第 2 例 N=4096 默认不跑**：会把设备卡死，必须显式写 case id） 2026-09-22 新增：数值正确（`out_err=2.44e-04`、`state_err=0`，第 3~9 例复测：B=8 变长 1.95e-03/0、短序列 1 ulp、null slot 0、D=320 pad、state_len=6 全过），但 **N=3072/4096 卡死设备**（N≤2560 通过）；根因 = 整条序列驻留 16MB VMEM（无 N 分块）；host 还拒绝 `idx=-1`（与现有 `_causal_conv1d` 的 pad slot 语义不同）。支持逐例执行，便于「一例一张干净卡」 |
| `probe_new_kda_conv_prefill_state_stride.py [dlc:N]` | #01 | 2026-09-22 晚新增、当晚升级（**发现的 P0**）：证明新算子的 `direct_state` 快路径**忽略 state stride**。四组对照（数据相同、只有 state 行距不同）：A 新算子+紧凑 `L*D` → `out_err=9.77e-04/state_err=0`；**B 新算子+池子行距 `L*D+2048` → `out_err=3.545/state_err=3.453`，且行间段 4096 个哨兵元素被改写**（写坏 recurrent 内存）；C 现有 `_causal_conv1d`+池子行距 → `out_err=3.91e-03/state_err=0`、哨兵零改写（同布局下现有算子是对的）；D 现有+紧凑 → 同上。dlc:0/dlc:1 输出一致，安全、不卡设备 |
| `probe_new_kda_conv_update.py` | #02 | 2026-09-22 新增：新算子 `dlc_kda_causal_conv_update` 逐元素正确（decode/spec/null slot/3D/2D state_indices）；`idx=-1` 行输出为脏值（`out_maxabs=1.71`，现有算子为全 0）；`D=320` 报「需 256 倍数」 |
| `probe_new_kda_conv_update_window.py` | #02 | 2026-09-22 新增：L=3/4/6（num_spec=0/1/3）下新算子 `state_err=0` 全过；已有 `_causal_conv1d_update` 在 `acc<T` 时全部不吻合（`state_err≈2.6~3.0`）→ 两者 spec 提交语义不同 |
| `probe_new_kda_conv_update_scaling.py` | #02 | 2026-09-22 新增、当晚加 warm-up 重测：槽位扫描（4/256/1024/1509/2048/3072）。真机几何 1509：新算子 **7.49ms** vs 已有 **0.59ms**（≈13×）；关键特征是**新算子耗时随槽位总数线性增长**（0.61→14.73ms）而现有算子与规模无关 → 证明成本是对整个 state 池的无条件搬运（**归因已由 `..._cost.py` 修正：94% 在 kernel 内的整池 DMA，host wrapper 只占 6%**；早期 70.3ms/110× 含首次初始化，已作废） |
| `probe_new_kda_conv_update_cost.py [dlc:N]` | #02 | 2026-09-22 深夜新增（**修正 #02 的根因归因**）：把「13×」拆成 host 与 kernel 两半 —— host 侧 4 遍整池拷贝实测只有 **0.47 ms（6.3%）**，**kernel 内整池 `HBM2VMem`（`kda_causal_conv_update.c:109`）≈ 6.9 ms（93.7%）**；B=1/8/64 → 7.59/7.45/7.50 ms（与活跃请求数无关），池子 2.21/6.63/13.26/26.53 MB → 1.29/2.29/4.02/7.39 ms（`t ≈ 0.74 + 0.251 ms/MB`，≈4 GB/s）。早期「成本来自 wrapper 整池拷贝」是按流量推算、非实测，已作废 |
| `probe_new_kda_conv_update_stride.py [dlc:N]` | #02 | 2026-09-22 深夜新增：对照 #01 的 state-stride P0 —— 紧凑行距 vs 池子行距（行间 2048 哨兵）两组都是 `out_err=0 / state_err=0 / committed_err=0 / 行间段被改写=0`，即 **update 算子没有该缺陷**（host 在 `CausalConv1d.cxx:401` 先 `transpose(1,2).contiguous()` 拉平，kernel 的紧凑寻址才成立；代价是 0.07 ms/层的整池拷贝）。若改零拷贝 strided，必须同时给 kernel 加行距元数据 |
| `probe_new_kda_chunk_prefill.py [--static] [dlc:N --wedge-device]` | #03 | 2026-09-22 新增、当晚改两段式：**默认 `--static` 只做不占卡的静态复现**（5 维张量用 4 阶 permute → `TensorShape.cpp:1417` 必抛异常）；加 `--wedge-device` 才做真机复现（抛异常 + **卡死设备** + `beta_logits` 被就地改写，`dlc_kda_chunk_prefill.cpp:416-418`）。本次真机复现把 `dlc:5` 打到需复位 |
| `probe_new_kda_recurrent_decode.py` | #04 | 2026-09-22 新增：新算子 `dlc_kda_recurrent_decode` **`out` 恒为全 0**（预填 7.0 被覆写成 0）；state 递推数值正确 |
| `probe_new_kda_recurrent_decode_fla_ref.py` | #04 | 2026-09-22 晚新增（替代作废的 `..._layout.py`）：按 fla 语义（`h=[S,H,Dk,Dv]`、衰减沿 Dk）逐项对齐 → `state_err≈3~4e-5`，T=1/2/4 与变长全部一致；`out` 逐位为 0（`(out==0).all()==True`，即 kernel `dlc_kda_recurrent_decode.cpp:189` 把常量 0.0f 写进 torch 回读的那半）；`num_accepted_tokens` 未生效（与「整窗提交」`3.0e-05` vs「accepted 快照」`2.6e-01`）；另含两例不依赖参考实现的受控判决（索引约定 / 衰减轴） |
| `probe_new_kda_rmsnorm_gate.py` | #06 | 2026-09-22 新增：新算子 `dlc_kda_rmsnorm_sigmoid_gate` **满足需求** —— sigmoid 语义 err 0.34%（swish 版差 15.0），3D/4D、weight bf16/fp32、T=1/T=0 全通过 |
| `probe_update_issue_repro.py [dlc:N]` | #02 | 2026-09-22 深夜新增（**一页式复现，推荐先跑这个**）：一条命令打印全部 4 个问题 —— `[P0-1]` slots=1509 新算子 **7.27ms** vs 现有 `_causal_conv1d_update` **0.20ms（35.8x）**、slots=64 仅 0.66ms（随槽位线性增长）；`[P0-2a]` 不 sync 调用 = 7.29ms（host 被 D2H 卡住）；`[P0-2b]` `DLCGraph` 捕获失败（Unknown synStatus）；`[P1-3]` host 4 遍拷贝 0.18ms/层；`[P1-4]` pad 行 `maxabs=1.73`（现有算子全 0，因 `at::zeros`）。dlc:6/dlc:2 结果一致，不卡设备 |
| `probe_kpool_pooled_read.py [dlc:N]` | #09 扩展（池读出，原 #08 读出侧） | 2026-09-22 新增：CPU 语义核对（批量版 vs native `read_pools` 逐元素一致 `max_err=0`、负 id→0、越界报错、dtype/空输入）+ 真机计时/对拍（独立读出算子已撤销，读出并入 `dlc_kpool_select_topk` 的 kernel 内地址算术；脚本保留用于核对语义与 native 逐请求循环耗时）。参考实现 `../refs/kpool_reference.py` |
| `probe_device_health.py [dlc:0 ...]` | 通用 | 2026-09-22 晚新增：逐卡体检（`matmul`+`sum`+`synchronize`，120s 超时）。`exit=0 ... OK` = 健康；`超时 → 需要复位` = 已卡死。卡死后**换卡**，复位后再验 |

> **2026-09-22 新增**：算子需求用的 **fp32 参考实现**（state 布局统一为 `[S,H,Dv,Dk]`）不在本目录，
> 在 [`../refs/kda_reference_dv_dk.py`](../refs/kda_reference_dv_dk.py)：`chunk_prefill_ref()`（#03）与
> `recurrent_decode_ref()`（#04），跑 `python ../refs/kda_reference_dv_dk.py --selftest`（纯 CPU，不占设备）
> 会自检「chunk 版 == 逐 token 递推」以及 decode 的 PAD slot / `num_accepted_tokens` 语义。

| `probe_kda_state_index_layout.py [dlc:N]` | #04/#02 | 2026-09-29 新增：`state_indices` 三种传法 —— `clone(block_table[:B,0])`（现状）/ `block_table[:B]`（`[B,1]` 二维切片，dlc-contiguous）/ 原始列视图 —— 对 #04 recurrent decode 与 #02 conv update 的结果一致性与单次耗时。三种传法**逐位一致**：二维切片无需改 ABI 即可免掉每层一次拷贝（单次拷贝被 ~0.2–0.4 ms 的 launch 掩盖，收益是每步少 34 次下发） |
| `probe_kda_decode_operand_views.py [dlc:N]` | #04 | 2026-10-09 新增：`_fused_operand` 的 9 个操作数逐个实测「模型侧 view 直传」是否可行。**硬失败（必须物化）**：`q/k/v`=split 切片 → `:72`（T>1 不连续）/`:76`（T=1 时 k/v 带 offset）；`raw_g`=`reshape([T,H*D])` → `:107` 512B 物理行（查的是**父 storage**描述符）；`beta`=合并 GEMM 切片 → `:72`/`:76`。**能过校验但每次调用被 launcher 隐式 staging**：`query_start_loc` +12.7 µs、`state_indices` 1-D 列视图 +11.2 µs / `[B,1]` 切片 +19.1 µs（B=64）；`A_log`/`g_bias` 根参数 reshape 则完全零开销。**附**：`B == 128` 会让 kernel 断言失败（metadata cache 只装 128 个 int32 却要读第 129 个），实测 126/127/129 正常 |

## 对应的回归测试

下面这些探针是**核查/定位用的最小复现**；其中 #02 的验收标准已固化为可直接跑的 pytest 回归：

```bash
cd /root/workspace/vllm-cl
DLC_TEST_DEVICE=dlc:6 /root/miniconda/envs/qingluo-dev1/bin/python -m pytest \
    tests/test_glm5_kda_conv_update_op.py -v      # 当前 7 passed / 4 xfailed
```

4 个 `xfail` 就是 #02 待修的四条（pad 行输出、无 host 同步、graph 可捕获、成本与槽位数无关），
算子组修好后会自动 `XPASS`，把标记去掉即验收通过。

## 注意

- 这些脚本是**核查用的最小复现**，不是回归测试；正式回归应放到 `vllm-cl/tests/` 下。
- **`probe_new_kda_chunk_prefill.py` 会把设备卡死**（算子缺陷，非脚本问题）：跑完必须复位该设备；
  本轮已因此打挂 `dlc:2`、`dlc:3`。`probe_new_kda_conv_prefill.py` 在 N=3072/4096 时同样会卡死，
  建议直接跳过该用例或只跑 ≤2560。
- **`dlc:0` 的 strided 整池拷贝异常慢**（26.5 MB ≈ 2.98 ms，`dlc:1~7` 约 0.04~0.07 ms；用户复位后依旧）：
  做 host 侧耗时对比请用 `dlc:1~7`，数值结论在 `dlc:0` 同样成立（算子本体耗时 8 张卡一致，约 7.4 ms）。
- **测例与设备纪律（2026-09-22 晚）**：一例一张干净卡，卡住的用例换卡重测并记为算子缺陷；被卡死的卡要复位后才能再测。
  本轮据此把 `probe_new_kda_conv_prefill.py` 改成可逐例执行；`dlc:0/2/3` 曾被卡死、复位后恢复。
  D 卡的复位由人工/驱动侧完成，脚本层无法自恢复。
- `#11` 的 512-NoPE 用例会让 kernel 卡死，请单独运行并准备好 `Ctrl-C`（必要时 `pkill -f probe_sparse_attention`）。
- `#18` 的对齐前提是「每块字节数 % 512 == 0」（kernel 用 `ALIGN128(page_size_el)` 作写入长度与 block 步长）；
  GLM-5.3-Flash 的 5 类页全部满足，所以 `probe_zero_block_ids.py` / `..._page_alignment.py` 里的
  `exact_match=False` 只是反例复现，不代表本模型有缺陷。
