# GLM-5.3-Flash MTP 端到端排查记录

日期：2026-11-10  
范围：`release-glm53flash` + TYD 真机 + MTP eager/graph（`num_speculative_tokens=1`）

结论：MTP eager 单请求 greedy 精度阻断已关闭；MTP + `FULL_DECODE_ONLY` graph
已完成单请求、in-flight 1/8/16/32/64 并发和 17K/34K/64.6K 长上下文语义验证。
发布前仍需补 AIME26、正式 acceptance/性能验收和更深 spec depth 验证。

## 1. 当前结论

**最终状态（2026-11-13）**：`g_a` 的隐式物化会进入 `custom_slice_16bit`
branch 5；旧实现经 CMEM staging，在连续 launch 下首个 64B 段可能残留前序数据。
已将 branch 5 改为独立的 VMEM 输入/输出分区并直接 `bf16_v2v_withoff ->
vmem_to_hbm`，真实 MTP split-view 50/50；最终无 MTP `FULL_DECODE_ONLY` graph
baseline 再次 20/20。模型侧 `index_select` workaround 已移除。

2026-11-13 进一步完成 MTP + `FULL_DECODE_ONLY` graph 验证：单请求 20/20，
并发语义请求 1/8/16/32/64 全部返回各自 marker，长上下文 17K/34K/64.6K
marker 全部命中。期间修复了 graph warmup spec metadata、KDA 非连续 indices、
spec batch padding 三个问题，详见 10.7。

本轮完成并验证了两处确定缺陷：

1. KDA causal-conv **prefill 历史窗口错位**；
2. KPool speculative transaction 在 prefill 阶段错误地按采样 token 数裁剪提交范围。

修复后，MTP 首个 32-token 请求可以逐 token 对齐 baseline，但连续请求仍会漂移。
失败不是 rejection sampler 的接受/拒绝判断问题，而是 **target forward 自身给出的
argmax 已经错误**。baseline 在同样请求下 20/20 稳定，因此问题限定在 MTP spec
路径。

## 2. 运行配置

基础启动方式：

```bash
cd /root/workspace/work-docs/glm53flash/vllm-cl/docs/glm5_flash

PORT=18005 \
MAX_MODEL_LEN=4096 \
GPU_MEMORY_UTILIZATION=0.85 \
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
bash start_glm5_flash_vllm_cl.sh \
  --eager \
  --no-enable-prefix-caching \
  --mamba-cache-mode none \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 1 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

排查时另外验证过：

- `--no-async-scheduling`；
- `DLC_KPOOL_SPECULATIVE_TRANSACTION=0`；
- `--mamba-cache-mode all`；
- KPool transaction 行级快照；
- KDA/KPool 元数据 page/slot/accepted 轨迹日志。

固定 oracle 请求：

```json
{
  "model": "GLM-5.3-Flash-Mix",
  "prompt": [100, 200, 300, 400, 500, 600, 700, 800],
  "max_tokens": 32,
  "min_tokens": 32,
  "temperature": 0,
  "ignore_eos": true,
  "stream": false,
  "return_token_ids": true
}
```

baseline 的稳定 greedy 输出为 6-token 循环：

```text
200, 300, 400, 500, 600, 700, ...
```

## 3. 已确认修复

### 3.1 KDA causal-conv prefill 历史窗口错位

spec 路径下 conv state 宽度为：

```text
state_len = conv_width - 1 + num_spec = 3 + 1 = 4
```

但宽度 4 的 causal conv 在 prefill 时只能把 **前 3 行**当作有效历史，第 4 行是
spec scratch。旧实现把整个 `state_len` 当历史窗口，prefill 后第一次 verify 会整体
偏移一格。

表现为：首个 token 仍正确，第二个 token 开始漂移；后续 spec step 又可能自行恢复，
因此很容易误判成 acceptance 或 rejection sampler 问题。

修复：

- `DLC_Custom_Kernel/dlc_kernels/conv/kda_causal_conv/kda_causal_conv_prefill.c`
  - 卷积历史只读取前 `KDA_HISTORY_LEN=3` 行；
  - state 更新只把最后 3 个 token 写入前 3 行；
  - spec scratch 行不再参与窗口计算。

新增回归：

- `vllm-cl/tests/test_glm5_kda_conv_update_op.py::test_spec_prefill_uses_first_three_history_rows`

验证：

- `state_len=4` prefill 后，前 3 行与最后 3 个 prompt token bit-exact；
- 紧接着执行 `accepted=1` 的 spec conv update，输出和 state 与参考实现均为 0 误差。

### 3.2 KPool transaction 的 prefill 提交范围

旧 transaction 对所有请求都按 `sampled_token_ids >= 0` 的数量裁剪提交范围。prefill
只有 1 个采样结果，导致 8-token prompt 的 KPool cache 实际只提交了 1 个 token。

修复：

- `vllm-cl/vllm_cl/kpool_speculative.py`
  - transaction 记录 `scheduled_spec_decode_tokens` 对应的请求行；
  - 只有真正带 draft token 的请求按 accepted 裁剪；
  - prefill / chunked prefill / 普通 decode 全量提交 query token。

同时把 pool-cache 快照从“整页”改为“具体 pool row”：

- pool slot 只快照 `(block_page, slot_in_page)`；
- 避免回滚整个 64-pool page 时误伤同页其他 slot，便于后续多请求场景。

### 3.3 测试基线修正

修正了两个测试自身的问题：

- spec recurrent graph 测试原先在 capture 后修改 `q`，却拿修改前的输出做参考；
- 补充 spec conv prefill 回归。

当前定向测试结果：

```text
44 passed, 1 warning
```

范围：

- `test_glm5_mtp_kpool_ops.py`
- `test_glm5_kpool_speculative_transaction.py`
- `test_glm5_kda_token_state_pages.py`
- `test_glm5_kda_conv_update_op.py`
- `test_glm5_kda_recurrent_decode_op.py`

## 4. 失败现象

### 4.1 不是 rejection sampler 错

调试打印显示 target argmax 在 rejection 之前就已经错误。例如：

```text
draft=[300] target=[500]
```

此时正确 target 应为 `300`。top-5 logprob 也显示正确 token 的裕量很大，不是并列
数值翻转。

因此应继续查 target forward 的 KDA/KPool/MLA cache 状态，而不是 rejection 规则。

### 4.2 典型错误序列

正确：

```text
200, 300, 400, 500, 600, 700, 200, 300, ...
```

观察到的主要错误形态：

```text
200, 500, 200, 300, ...
200, 300, 200, 300, ...
200, 411, 200, 411, ...
```

错误通常固定在输出第 2、3 或第 8 个 token 附近，随后序列会沿错误 token 继续，
并不是一次孤立数值误差。

### 4.3 baseline 稳定

同一服务配置、同一固定请求、无 MTP：

```text
32-token × 20 次：20/20 全对
```

所以模型权重、tokenizer、采样参数和基础 KDA/KPool 路径不是主要噪声源。

## 5. 实验矩阵

以下为固定 32-token greedy 请求的观察结果。不同服务重启后的 block 分配、page ID
和缓存残留会影响首次失败位置，因此数字用于判断方向，不作为最终性能/精度指标。

| 配置 | 通过数 | 备注 |
|---|---:|---|
| baseline，无 MTP | 20/20 | 稳定 |
| MTP，默认 async，transaction on | 15/20 | 失败集中在第 2/3/8 token |
| MTP，`--no-async-scheduling`，transaction on | 18/20 | 失败率下降，但仍未消除 |
| MTP，默认 async，transaction off | 20/20（32-token） | 32-token 曾全过 |
| 同上，128-token | 5/5 | 短中序列稳定 |
| 同上，256-token | 2/5 | 长序列后再次出现漂移 |
| MTP，`--no-async-scheduling`，transaction off | 16/20 | 仍失败 |
| `mamba-cache-mode=all`，transaction on | 18/20 | 仍有同样问题 |
| `mamba-cache-mode=all`，transaction off | 14/20 | 不能仅靠 mode 修复 |

说明：

- `--no-async-scheduling` 能降低失败率，但不是根因；
- 关闭 KPool transaction 能让 32/128-token 短中序列通过，但 256-token 后仍不稳定；
- `mamba-cache-mode=none/all` 都不能消除问题。

## 6. 已排除项

### 6.1 KDA conv 算子多步 accepted 语义

用 `accepted = 1,2,2,1,1,2` 进行多步对拍，conv 输出和 state 与参考实现完全一致。

### 6.2 KDA recurrent per-token page 写回

使用 `[p0,p1]`、`[p0,p0-1]`、低页和高页（含 `[68,67]`）做多步对拍：

- 初始页按 `accepted-1` 选择正确；
- token0/token1 分别写入各自 page；
- output 和 state 均与参考一致。

### 6.3 prefill/scatter 对脏页的覆盖

对 page 4、67、68 注入随机脏 state，再执行真实 chunk-prefill + state-scatter：

- prefill 输出一致；
- prefill 后 state 一致；
- 高页号没有地址越界或未覆盖问题。

### 6.4 KPool 有效范围

对 zero pool 与 random stale pool 使用同样的 prompt 8-token 输入：

- 有效 pool page 0/1 都被正确覆盖；
- stale pool 只存在于当前 sequence 不可见范围；
- `#09` 的 per-query tail/pool 选择对已完成的未来 pool 有正确长度掩码。

### 6.5 page ID 不是低层算子的直接根因

某轮失败请求反复拿到 `spec_state_indices=[[68,67]]`，通过请求为 `[[4,5]]`。但将
同样的 page 序列直接喂给 conv/recurrent 多步 kernel 时均正确。因此 page ID 相关性
更可能指向 **上层 cache 复用/请求间残留/共享 buffer 顺序**，而不是简单页号溢出。

## 7. 当前最可疑方向

按优先级排列：

1. **KPool cache 在多步 spec 与请求复用之间的状态一致性**
   - transaction 开关会显著改变失败模式；
   - 32/128-token 可过、256-token 失败，说明可能累积；
   - 需要在每个 target step 前后对 pool/tail cache 做 checksum，并与“只 append
     accepted prefix”的参考路径逐步比较。

2. **MTP draft 与 target 共享 `topk_indices_buffer` 的顺序问题**
   - checkpoint 配置 `index_share_for_mtp_iteration=true`；
   - proposer 会把 target 的 topk buffer 提供给 draft；
   - draft 在 target sampling 后运行并执行 `compact_topk_indices`；
   - 需要在关闭 index share 或隔离 buffer 的条件下对比失败率。

3. **请求/block 复用后的缓存残留**
   - 长请求后重新跑短请求仍可能失败；
   - 失败请求的 KDA/KPool page 组合与通过请求不同；
   - 需要在新请求开始时清零对应 KPool/KDA 状态块，或证明所有有效位置都被覆盖。

4. **async scheduling 仅放大问题**
   - `--no-async-scheduling` 从约 25% 失败降到约 10%；
   - 仍可复现，说明还存在非 async 路径的缓存错误。

## 8. 建议下一步

1. 加“逐步 KPool cache 对拍”探针：
   - 每 step 打印/保存 pool cache、tail cache、slot mapping、accepted；
   - 对比 transaction on/off 与只 append accepted prefix的参考；
   - 重点检查 `pool boundary`、`accepted` 从 2 变 1、第二次回到首 token 的位置。

2. 隔离 MTP index sharing：
   - 临时禁用 `index_share_for_mtp_iteration`；
   - 或为 draft 使用独立 topk buffer，验证 target argmax 是否恢复。

3. 做“新请求强制干净 block/page”实验：
   - 在 request 开始前清零其 KPool tail/pool page 与 KDA state pages；
   - 若失败消失，转为补 request 生命周期清理，而不是继续调 kernel。

4. 稳定后再推进：
   - 32/128/256 token 连续单请求；
   - batch=8/32；
   - prefix cache；
   - `FULL_DECODE_ONLY` graph；
   - 长上下文与 acceptance/吞吐指标。

## 9. 关键文件与日志

已修改/新增：

- `DLC_Custom_Kernel/dlc_kernels/conv/kda_causal_conv/kda_causal_conv_prefill.c`
- `DLC_Custom_Kernel/dlc_kernels/conv/kda_causal_conv/kda_causal_conv_update.c`
- `DLC_Custom_Kernel/dlc_kernels/attention/kda/dlc_kda_recurrent_decode.cpp`
- `vllm-cl/vllm_cl/kpool_speculative.py`
- `vllm-cl/vllm_cl/worker/dlc_model_runner.py`
- `vllm-cl/tests/test_glm5_kda_conv_update_op.py`
- `vllm-cl/tests/test_glm5_kda_recurrent_decode_op.py`

代表性日志：

- baseline：`/root/workspace/log/baseline_repeat_20261110.log`
- KDA/KPool 轨迹：`/root/workspace/log/mtp_smoke_20261110_kdatrace.log`
- no-async：`/root/workspace/log/mtp_smoke_20261110_noasync.log`
- no-async + transaction off：`/root/workspace/log/mtp_smoke_20261110_noasync_notx.log`
- mamba all：`/root/workspace/log/mtp_smoke_20261110_mambaall.log`
- 最新安装 kernel：`/usr/local/chipltech/synapse/lib/libcustom_dlc_perf_lib.so`

当前服务状态：已停止；未提交 commit。

## 10. 2026-11-12 更新：MTP target verify 仍是阻断项

### 10.0 排查逻辑链

本轮按“先缩路径、再缩算子、最后缩时序”的顺序推进：

1. **先确认 baseline 没问题。**
   同一模型、同一 kernel、同一启动脚本关闭 MTP，固定请求 20/20 稳定；
   因此不能把问题直接归因到 KDA/KPool 的通用实现。

2. **确认 KPool transaction 是否是放大器。**
   对比 transaction on/off。off 仍有失败，说明事务不是唯一根因；但事务在
   all-accepted 时也会多余 replay，因此先收紧为“只有 rejected row 真正完成
   pool 才回滚”，并补单测。

3. **排除 MTP index-sharing / top-k compaction。**
   关闭 `index_share_for_mtp_iteration`，以及让 target/draft 使用独立
   `topk_indices_buffer`，失败率没有本质变化，排除这两条共享逻辑。

4. **区分 draft forward 与 target verify。**
   增加诊断开关，跳过 draft model forward，只返回固定 draft token；target
   仍走 2-token speculative verify。失败继续存在，说明根因在 target verify
   前向，而不是 draft cache/通信副作用。

5. **判断是否为普通异步调度竞争。**
   用 `DLC_SYN_BLOCKING=1` 串行化 host 调用。失败形态改变但未消失；随后又
   分别关闭 custom allreduce、在 `#07 -> #09` 之间显式 sync，均不能单独修复。

6. **排除 mamba 状态模式。**
   `mamba-cache-mode=none/all/align` 都复现失败，排除单一的 cache 复用模式。

7. **回到算子层逐项对拍。**
   - KDA conv/recurrent 多步 accepted=1/2 切换：通过；
   - KDA 高 state slot 68：通过；
   - KPool `#09` T=2、非零 history、production D/H/topk：通过；
   - KPool `#07` 双 token tail 写入：通过；
   - KPool transaction all-accepted/tail-only/pool-close：通过。
   这说明基础算子契约不是当前首要嫌疑。

8. **验证高 HBM 地址假设。**
   MTP 额外 draft/cache 会把 target cache 推到更高地址。临时先分配 20 GiB
   dummy，再跑高 KDA page / 高 KPool pool-slot，单算子仍通过，排除简单地址
   溢出。

9. **验证跨 launch 的 XYS 完成顺序。**
   连续发起 16 组 KDA recurrent kernel，中途不做 host sync，最后一次同步后
   统一校验；结果全部通过，排除“单纯 back-to-back launch 未串行化”。

10. **当前收敛点。**
    失败只存在于 target 的 2-token speculative verify；表现为两类：
    首次 spec step 即漂移，以及跨 pool 边界后漂移。下一步需要对比同一请求
    在 baseline row0 与 MTP verify row0 的逐层 hidden/logits，定位第一个
    不一致的算子簇，而不是继续扩大配置矩阵。

### 10.1 本轮代码改动

KPool speculative transaction 已收紧为“只有被拒绝且实际完成 pool 的行才回滚”：

- 全接受或普通 decode/prefill：不做 restore/replay；
- 拒绝只污染 raw-tail、但没有完成 pool：下一位点会覆盖对应 tail slot，不做回滚；
- 拒绝行带有非负 `pool_slot`：snapshot/restore + accepted-prefix replay 仍然执行。

同时修复了 `record()` 在构造 `_Call` 后才计算 `pool_slots_cpu` 的局部变量错误，并补充：

- all-accepted / rejected-tail-only / rejected-pool-close 三类事务判定测试；
- 拒绝 tail 行在下一 pool close 前被覆盖的算子级测试；
- KDA 高 state slot（68）的 spec prefill + update 回归测试。

定向测试：

```text
vllm-cl/tests/test_glm5_kpool_speculative_transaction.py   11 passed
vllm-cl/tests/test_glm5_mtp_kpool_ops.py                   通过
vllm-cl/tests/test_glm5_kda_conv_update_op.py::test_spec_prefill_and_update_use_high_state_slot
                                                           passed
```

### 10.2 实验矩阵

固定请求仍为 8-token prompt + 32-token greedy 输出，服务为 eager、TP8、
`num_speculative_tokens=1`。无 MTP baseline 与 MTP 使用相同模型、kernel 和
启动脚本。

| 配置 | 结果 | 结论 |
|---|---:|---|
| 无 MTP baseline | 20/20 | 基础 KDA/KPool/MLA 路径稳定 |
| MTP，transaction off | 17/20 | 关闭事务不能修复 |
| MTP，index-share off | 18/20 | `compact_topk_indices` 不是唯一根因 |
| MTP，target/draft 独立 top-k buffer | 19/20 | 共享 top-k buffer 不是唯一根因 |
| MTP，`DLC_SYN_BLOCKING=1` | 19/20 | 串行化只改变失败形态，仍有漂移 |
| MTP，`mamba-cache-mode=all` | 17/20 | 不修复 |
| MTP，`mamba-cache-mode=align` | 17 次内已多次失败 | 不修复 |
| MTP，跳过 draft forward，仅返回固定 draft token | 13/20 | 问题在 target 的 2-token verify 前向，不是 draft 副作用 |
| MTP，`--disable-custom-all-reduce` | 46/50 | 有所改善，但不能消除 |
| MTP，`#07 -> #09` 之间显式 device sync | 前 32 次已多次失败 | 简单全同步不是正确修复 |

代表性失败：

```text
[200, 200, 200, ...]                 # 第一次 spec step 即漂移
[200, 411, 200, 411, ...]            # 第一次 spec step 即漂移
[200, 300, 400, 500, 600, 700,
 200, 500, 200, 300, ...]            # 第 7 个输出、跨 pool 边界后漂移
```

### 10.3 已排除的算子/配置项

- KDA conv/recurrent 单算子多步：accepted 在 1/2 间反复切换，状态和输出均与
  FP32 reference 一致；KDA page `68/67` 与 `4/5` 均通过。
- KDA 高 state slot 的 prefill + spec update：通过。
- KPool `#09`：T=2、`before_len=8..9`、production D=128/H=32/topk=2048 与
  reference 逐位一致。
- KPool `#07`：双 token raw-tail 写入与逐 token 调用逐位一致。
- KPool transaction：all-accepted、tail-only reject、pool-close reject 的
  snapshot/replay 结果与 accepted-prefix reference 一致。
- `mamba-cache-mode=none/all/align`、index-share、独立 top-k buffer、custom
  allreduce、`#07/#09` 显式同步均不能单独解决。

### 10.4 当前判断

失败至少包含两类：

1. **2-token target verify 的早期漂移**：同一组输入下，layer 0 的 KDA
   conv/recurrent state checksum 与正常请求一致，但 layer 1 起的完整输出开始
   分叉；跳过 draft forward 后仍复现，因此不是 draft 的 cache 副作用。
2. **跨 pool 边界的漂移**：通常从第 7 个输出附近开始，说明 KPool
   transaction/append-select 仍有未覆盖的模型级时序或 cache 一致性问题。

因此当前不能把 MTP 标记为精度验收通过，也不能用“单算子测试全过”替代模型级
结论。下一步应把失败请求的 16 个 target step 输入 token、position、
query_start_loc、KDA state page、KPool slot mapping 与 logits argmax 一次性导出，
优先复现“第一次 spec step 即漂移”的最小模型级 case；之后再单独收敛 pool 边界
失败。

### 10.5 进一步细粒度定位

随后继续按“层 -> self-attn 子阶段 -> TP rank”细分：

1. **逐层 bitwise hash**

   对 decoder layer 0–5 的输入 hidden/residual 做 BLAKE2b hash。正常请求和
   失败请求在同一步对比时：

   - prefill 后 layer 0–5 全部一致；
   - 第一次 spec step 的 layer 0 输入/输出一致；
   - **layer 1 输入一致，但 layer 1 输出 hash 第一次不同**。

   首个确定分叉点在 layer 1，而不是 layer 0、prefill 或 draft。

2. **layer 1 self-attn 子阶段**

   在 layer 1 内分别记录 `x0 -> self_attn -> hc_post_pre -> MoE` 的 hash。首个
   不同点是 `self_attn` 输出。再拆 KDA 路径：

   - KDA recurrent core 输出 hash 在 run-to-run 中相同；
   - `o_norm` 输出 hash 在部分 rank/step 先不同；
   - `o_proj` 对同一输入重复调用是 bitwise 确定的。

3. **跨 TP rank 对比**

   对 8 个 TP rank 分别记录 KDA core/o_norm hash。失败样本中：

   - 多数 rank 的 KDA core 完全一致；
   - 部分 rank 的 `g2 = g_b_proj(g_a)` 不确定，部分 rank 的 `o_norm` 不确定；
   - `g_a` 在早期 step 一致，但在后续 step 会因前一步 hidden 分叉而全部变化。

   说明不是 rank 0 单点的 fused kernel 问题，而是 layer 1 的 gate/projection 与
   fused RMSNorm 路径中存在跨 rank 的状态不确定性。

4. **fused o_norm fallback**

   用纯 PyTorch `rsqrt(mean(x^2)+eps)` + sigmoid gate 替换
   `custom_kda_rmsnorm_sigmoid_gate_bf16` 后，MTP 仍为 `16/20`，说明该 fused
   kernel 不是唯一根因；但它仍是必须继续核查的路径之一。

5. **发现的真实 kernel 健壮性问题**

   `custom_kda_rmsnorm_sigmoid_gate_bf16` 对输入 storage offset/地址对齐敏感：
   同一组数据放到非 128-byte 对齐的 storage offset 时结果会变化。模型当前使用
   512-byte 对齐 allocation，因此这不是本轮 MTP 失败的直接解释。已在
   `FusedRMSNormGatedDLC` 中对未对齐的 core/gate 视图做条件 clone，并补回归
   `test_rmsnorm_sigmoid_gate_handles_unaligned_storage`。

6. **本轮额外排除项**

   - oracle draft（强制全接受）仍失败，排除 rejection/accepted 语义；
   - 每 decoder layer 后全局 sync 仍失败，排除简单层间未同步；
   - `USE_CPU_FOR_COLLECT_OP=True` 仍失败，排除单一设备 collective 算法；
   - KDA recurrent 同输入/state 的 32 次 back-to-back 调用 bitwise 一致；
   - 20 GiB 高地址后的 KDA conv/recurrent、KPool #07/#09 单算子仍正确。

当前最小确定结论：**首个 bitwise 分叉发生在 layer 1 self-attn 的
`g2/o_norm/o_proj` 链路，KDA recurrent core 本身未先分叉**。下一步应导出该
step 所有 rank 的 `g_a/g1/g2/core/o_norm` 完整张量做 bitwise diff，重点检查
`g_b_proj` 的 replicated shard、`FusedRMSNormGatedDLC` 的 buffer 生命周期，以及
`RowParallelLinear` allreduce 前后的 tensor alias。

以上是当时的中间结论；后续张量级定位与最终修复见 10.6。

### 10.6 根因收敛与修复（2026-11-12）

#### 10.6.1 首个差异的精确位置

对失败请求的 layer 1 同时导出 `projected`、`g_a`、`g1`、`g2`、KDA core 和
`o_norm`，并在 `in_proj_qkvbfg_a` 之后立即做一次 `projected` 快照。失败样本中：

- `projected_early` 与最终 `projected` 逐字节一致；
- `g_a` 在失败请求和通过请求之间逐字节一致；
- KDA `core` 逐字节一致；
- 只有 `g2 = g_b_proj(g_a)` 的 **row 0** 出现约 `1021/1024` 个元素不同，
  row 1 完全一致。

用 checkpoint 中 layer 1 的 `g_b_proj.weight` 做 CPU BF16/FP32 参考，可逐位复现
通过路径的 `g2`，确认不是权重、replicated shard 或 RMSNorm 的问题。

#### 10.6.2 反解出的错误输入

`g2` 的每个输出列都是 `g_a[row] @ W[:, out]`。对失败 row 0 做最小二乘反解：

```text
x_effective[32:128] == g_a[row0][32:128]   # 误差约 1e-3，BF16 舍入量级
x_effective[0:32]   != g_a[row0][0:32]     # 旧值；输出预清零时为 0
```

也就是说，GEMM 的 K 维前 32 个 BF16 元素读到的是旧数据。由于这 32 个元素参与
所有 1024 个输出列，最终表现为整行 `g2` 都不同，而不是只有 32 列不同。同一次
调用中 row 1 未出现该问题，说明缺陷与具体行/XYS 路径相关。

#### 10.6.3 缺陷所在

`g_a` 是融合投影 `projected` 的最后一个分片：

```text
projected = [q/k/v | beta | f_a | g_a]
g_a = projected[:, 3208:3336]   # [T, 128]，stride=(3336, 1)
```

该 strided view 交给后续 column-major GEMM 前，DLC launcher 会通过
`custom_slice_16bit` 做隐式物化。模型级失败可反解出该物化结果的前 32 个
BF16 元素仍是旧值，但在 MTP 2-token verify 的模型路径上才稳定出现。

目前已为 `{3336,2} -> {128,2}, start=3208` 的精确形状补了独立 syntest，
并做了 2000 次连续下发 stress（后接 consumer copy），均为 0 失败；但新的
**两阶段耦合测例**已经稳定复现：

- phase A：固定 seed，走 `in_proj -> index_select(g_a) -> g_b`，保存正确 `out`；
- phase B：同一输入，走 `in_proj -> split(g_a) -> g_b`，触发隐式
  `custom_slice_16bit` staging，再与 phase A 逐字节比较。

结果（`T=2`，每轮 1000 次）：

| staging 方式 | 失败轮数 | `g_a_input` 差异 | 输出差异 |
|---|---:|---:|---:|
| 原始 split view | 211/1000 | 0 | row0 `1021/1024` |
| 显式 `.contiguous()` | 282/1000 | 32 个元素 | row0 `1021/1024` |
| `index_select` | **0/1000** | 0 | 0 |
| 输出预清零 + split view | 239/1000 | 0 | row0 仍异常 |

`T=1`、`T=8` 的 view/contiguous 在 500 次内均为 0 失败；`T=3..7` 的 view/
contiguous 也能触发失败，因此不是只限 `T=2`，但 `T=1` 的纯 decode 不触发。

这组结果说明缺陷确实在 **slice/copy 物化本身**，不是后续 GEMM 或输出 buffer：
显式 `.contiguous()` 也会产生 32 个错误的 `g_a` 元素，而 `index_select` 完全
绕开该 copy 路径后稳定正确。

进一步对首个失败样本做逐元素比对：`g_a_input[row0][0:32]` 与
`hidden[0:32]` **逐字节完全一致**，而 `projected` 和原始 `g_a` view 都正确。
将 `projected` 做 CPU roundtrip 后再复制仍失败；`.contiguous()` 前加
`dlc.synchronize()` 也仍失败。这说明不是 in_proj 写入未完成或 GEMM 输出未写，
而是 **slice staging 的 VMEM/CMEM 首段没有覆盖，残留了前序 in_proj 使用过的
hidden 数据**。

#### 10.6.4 已落地修复

在 `vllm/models/glm5next/common/kda.py` 中，不再把 `g_a` split view 直接交给
`g_b_proj`，而是用独立的 `index_select` gather kernel 物化：

```python
self._g_a_index = torch.arange(
    g_a_offset,
    g_a_offset + self.head_dim,
    dtype=torch.int64,
    device=projected.device,
)
g_a_for_proj = projected.index_select(1, self._g_a_index)
g_proj_states = self.g_b_proj(g_a_for_proj)[0]
```

索引张量按 layer/device 缓存。debug dump 中 materialize 结果与原 split view
逐字节一致（968 份样本，0 差异），只绕过了有缺陷的 slice copy，不改变数值。
耦合测例 `work-tests/mtp/run_slice_staging_compare.py` 的
`index_select` 模式在 T=1/2/8 各 500/1000 次均为 0 失败。

#### 10.6.5 验证结果

| 验证 | 结果 |
|---|---:|
| MTP eager，固定 8-token prompt，32-token greedy，重复 50 次 | **50/50** |
| 无 MTP baseline，FULL_DECODE_ONLY graph，重复 20 次 | **20/20** |
| 两阶段耦合：T=2 split-view staging / 1000 次 | **211 failed** |
| 两阶段耦合：T=2 explicit contiguous / 1000 次 | **282 failed** |
| 两阶段耦合：T=2 sync-before-contiguous / 1000 次 | **343 failed** |
| 两阶段耦合：T=2 CPU-roundtrip + contiguous / 1000 次 | **326 failed** |
| 两阶段耦合：T=2 index_select / 1000 次 | **0 failed** |
| KPool transaction + KDA conv/update + 融合算子定向回归 | **41 passed** |
| MTP KPool ops + DLC contract + token state + recurrent decode | **36 passed** |

关键日志：

- MTP 50/50：`/root/workspace/log/mtp_final_20261112.log`
- MTP 复核 50/50：`/root/workspace/log/mtp_final_verify2_20261113.log`
- 无 MTP graph 20/20：`/root/workspace/log/mtp_final_baseline_graph_20261112.log`
- `g_b` 历史失败复现：`/root/workspace/log/mtp_retest_projearly_20261112.log`
- `index_select` 首次验证：`/root/workspace/log/mtp_retest_gbindexselect_20261112.log`
- 两阶段耦合测例：`/root/workspace/work-tests/mtp/run_slice_staging_compare.py`
- 耦合测试结果目录：`/root/workspace/work-tests/mtp/results/slice_staging_compare/`

#### 10.6.6 后续事项

本节当时记录的是模型侧 `index_select` 定向绕过。后续已用两阶段耦合测例
（`work-tests/mtp/run_slice_staging_compare.py`）复现并定位到
`custom_slice_16bit` branch 5，完成 kernel 根修；模型侧 workaround 已移除，
最终回归见 10.6.8。MTP graph、并发、长上下文与 AIME26 仍属于后续发布验收项。

#### 10.6.7 `g_a` 物化/staging 路径逐项

从 Python 到最终 GEMM，当前实际路径是：

1. KDA forward 中 `projected = in_proj_qkvbfg_a(hidden_states)`；`g_a` 是
   `projected[:, 3208:3336]`，stride 为 `(3336, 1)`，storage offset 为 3208。
2. `torch.dlc_matmul_t_col_major_(out, g_a, weight)` 进入
   `dlc_matmul_t_col_major_impl`；此时 `left_trans=false`、`right_trans=true`。
3. `KernelDesc::input()` 发现 `g_a` 不是 DLC-contiguous，调用
   `to_contiguous_dlc()` → `resolve_copy()`。
4. `copy_plan` 通过 `detect_multi_slice_chain()` 还原出父张量
   `[3336, 2]` + slice(dim=1, start=3208, step=1)，最终进入
   `copy_by_slice()`；日志中对应 `custom_slice_16bit`。
5. `custom_slice_16bit` 收到 `slice_dim=0, start=3208, step=1, h=2`，走
   “`slice_step==1 && sliced_size_pad+256 <= vmem_size`”分支：
   `hbm_to_vmem_strided` → `bf16_offmemcpy(off_numel=136)` → `cmem_to_hbm`，
   并把两行拆到两个 XYS。这里的首 64B 正好对应 32 个 BF16 元素。
6. 物化完成后，`_set_dlc_tensor_impl()` 设置 GEMM 输入描述符并下发
   `custom_mm_ah_8_rhsT_bf16_col_major`；两者之间没有显式
   `dlcDeviceSynchronize()`，只有 `wait_for_async_copy()` 的 host-side copy
   等待。

因此优先怀疑以下位置：

- `custom_slice_16bit` 的 CMEM 分支中 `bf16_offmemcpy` 写入与
  `cmem_to_hbm` 读取之间的 FXC/DMA 可见性；
- `copy_by_slice()` 返回后，隐式 staging 与消费者 GEMM 之间没有 device-level
  完成栅栏；
- `[3336,2]` 父布局到 `[128,2]` staging 布局的 padding/offset 转换。

该节是定位阶段的中间推断；下面的两阶段耦合测例确认了
`split view -> contiguous/staging` 会在 T=2 等形状产生 32/64 个错误元素，
而 `index_select` 不经过该路径时完全正确。下一步应把失败样本中的
`g_a` staging buffer 和 `bf16_offmemcpy` 的 CMEM 内容直接导出。首个失败样本
已显示错误 32 元素等于 producer 输入 `hidden[0:32]`，优先怀疑
`hbm_to_vmem_strided`/`bf16_offmemcpy` 没有覆盖 VMEM 首段，导致前序 kernel
的数据残留。该推断已由 10.6.8 的 branch 5 修复和回归验证证实。

#### 10.6.8 kernel 修复与回归（2026-11-13）

已在 `DLC_Custom_Kernel/dlc_kernels/copy/slice_16bit.cpp:187` 的 branch 5 做修复：

- 原路径：`hbm_to_vmem_strided -> bf16_offmemcpy(CMEM) -> cmem_to_hbm`
- 新路径：`hbm_to_vmem_strided -> bf16_v2v_withoff(VMEM) -> vmem_to_hbm`

同时将 VMEM 输入/输出分区从 `ctx.vmem_size / 128` 调整为
`ctx.vmem_size / 2 / 128`，让偏移复制和回写使用独立 VMEM 区域，不再依赖
CMEM staging 的首段写入结果。新 kernel 已编译并安装到
`/usr/local/chipltech/synapse/lib/libcustom_dlc_perf_lib.so`，旧库备份为
`libcustom_dlc_perf_lib.so.bak-pre-slicefix-20261113`。

回归结果：

| 验证 | 结果 |
|---|---:|
| 耦合测例 T=2，view/contiguous/zero_out/index_select 各 1000 次 | 全部 **0 failed** |
| 耦合测例 T=1..8，view/contiguous | 全部 **0 failed** |
| `slice_bf16` syntest | **43/43 passed** |
| `slice_mtp_stress`，2000 次连续下发 | **0 failed outputs/elements** |
| MTP eager，固定请求 50 次 | **50/50** |
| 无 MTP baseline，`FULL_DECODE_ONLY` graph，固定请求 20 次 | **20/20** |

关键日志：

- MTP：`/root/workspace/log/mtp_slicefix_verify_20261113.log`
- 无 MTP graph baseline：`/root/workspace/log/baseline_slicefix_graph2_20261113.log`
- slice syntest：`/tmp/slice_bf16_after_fix.log`、`/tmp/slice_stress_after_fix.log`
- 耦合结果：`/root/workspace/work-tests/mtp/results/slice_staging_compare/`

无 MTP graph 回归使用 `GPU_MEMORY_UTILIZATION=0.80`、`max-num-batched-tokens=1024`、
`max-num-seqs=1`、未启用 MTP；20 次固定请求全部输出
`200,300,400,500,600,700,...`，测试后服务已停止。

修复后已临时关闭 `index_select`，让 KDA 直接使用旧 `g_a` split view；新 kernel
在真实 MTP 模型上再次通过 **50/50**。随后已移除模型侧 `index_select` workaround，
生产路径现在直接走修复后的 `custom_slice_16bit`。旧的 `index_select` 记录仅作为
定位历史保留。

split-view 模型日志：`/root/workspace/log/mtp_splitview_fixedkernel2_20261113.log`。

本轮日志：

- baseline：`/root/workspace/log/baseline_retest_20261112.log`
- transaction off：`/root/workspace/log/mtp_retest_notx_20261112.log`
- index-share off：`/root/workspace/log/mtp_retest_noindexshare_20261112.log`
- 独立 top-k：`/root/workspace/log/mtp_retest_isolatedtopk_20261112.log`
- blocking：`/root/workspace/log/mtp_retest_blocking_20261112.log`
- 跳过 draft：`/root/workspace/log/mtp_retest_skipdraft_20261112.log`
- 关闭 custom allreduce：`/root/workspace/log/mtp_retest_nocustomar_20261112.log`
- KDA layer checksum：`/root/workspace/log/mtp_retest_layerdiv_20261112.log`
- layer 全 hash：`/root/workspace/log/mtp_retest_hash_20261112.log`
- layer 1 子阶段：`/root/workspace/log/mtp_retest_l1stages_20261112.log`
- 跨 rank KDA：`/root/workspace/log/mtp_retest_l1cases_20261112.log`
- fused norm fallback：`/root/workspace/log/mtp_retest_onormfallback_20261112.log`

固定请求重复脚本：`work-tests/mtp/run_mtp_greedy_repeat.py`。

### 10.7 MTP + FULL_DECODE_ONLY graph 验证（2026-11-13）

#### 10.7.1 本轮发现并修复的问题

1. **Graph warmup 把 MTP verify batch 误判成 prefill**

   `FULL_DECODE_ONLY` warmup 的 dummy MTP batch 每请求 2 token，但没有经过
   `_prepare_inputs()`，`num_decode_draft_tokens` 未初始化。GDN metadata 因此把
   verify 行归类成 prefill，触发 `gather_initial_states`，在初始化阶段报
   `KDA state: indices must be a same-device contiguous int32/int64 vector`。
   已在 `vllm-cl/vllm_cl/worker/dlc_model_runner.py::_dummy_run()` 中为
   `uniform_decode + speculative_config` 的 warmup/capture 路径填充 dummy
   `num_decode_draft_tokens` 和 `num_accepted_tokens`。

2. **KDA gather/scatter 没有归一化非连续 indices**

   MTP 打开后 block table 变成多列，`block_table[:, 0]` 是 strided view。
   `patch_kda.py` 的 `gather_initial_states_dlc()` / `scatter_states_dlc()`
   现在会在设备、dtype 或连续性不符合时转换为同设备连续 int32，再调用
   `custom_kda_state_gather/scatter`。

3. **Spec graph padding 未覆盖 padded token 尾部**

   MTP 请求完成时间不同，target graph 会从 8 request 缩到 6 request，并把 token
   维 padding 到 16。`GDNAttentionMetadataBuilder` 的 spec 路径原先把
   `spec_query_start_loc` 尾部填成 12，但 KDA recurrent kernel 要求 cumulative
   boundary 到达 padded 的 16，导致 `synErrorLaunchFailure`。已在
   `vllm/vllm/v1/attention/backends/gdn_attn.py` 中把尾部 padding token 均分到
   末尾 NULL state slots；对应回归测试为
   `test_full_cudagraph_spec_metadata_covers_padded_tokens`。

#### 10.7.2 单请求与并发结果

最终配置：TP8、`max-model-len=66560`（长上下文）或 4096（并发语义）、
`max-num-seqs=64`、`DLC_SYN_GRAPH_CACHE=1`、MTP `num_spec=1`、
`FULL_DECODE_ONLY`。单请求固定 greedy 请求 **20/20**。

并发语义测试使用每请求唯一 marker，要求响应包含自己的 marker：

| in-flight 并发 | 轮次 | 通过 | wall 时间 | 输出吞吐 |
|---:|---:|---:|---:|---:|
| 1 | 2 | 2/2 | 1.34–1.42 s | 16.9–17.9 tok/s |
| 8 | 2 | 16/16 | 3.00–3.12 s | 61.6–64.1 tok/s |
| 16 | 2 | 32/32 | 3.89–3.95 s | 97.2–98.8 tok/s |
| 32 | 2 | 64/64 | 8.44–8.59 s | 89.4–91.0 tok/s |
| 64 | 2 | 128/128 | 17.57–19.72 s | 77.9–87.4 tok/s |

说明：受 KDA state/KV cache 容量限制，4K 配置下实际同时 `Running` 峰值约
**20**；32/64 档是 64 个 in-flight 请求排队执行，不等价于 64 个请求同时 decode。
日志确认 MTP proposer graph 在 reqs=1/4/8/12/16/20 上真实 replay；因此本轮验证了
排队/缩批/补 padding 路径，但没有覆盖 32/64 同时 decode 的 graph bucket。

#### 10.7.3 长上下文结果

随机 token 长 prompt 的 greedy 输出在 no-MTP baseline 自身也会跨服务/重跑分叉
（16K/64K 均观察到），因此逐 token 全等不适合作为长上下文验收判据。改用自然
文本 needle marker 检索，MTP eager 与 MTP graph 均全部命中：

| prompt tokens | MTP eager | MTP graph | marker |
|---:|---:|---:|---|
| 17,025 | 44.98 s | 44.60 s | NEEDLE-1000 |
| 34,025 | 92.09 s | 91.55 s | NEEDLE-2000 |
| 64,625 | 184.30 s | 184.06 s | NEEDLE-3800 |

64K 档使用 3800 行（64,625 token），因为 4000 行会达到 68K，超过
`max-model-len=66560`。MTP graph 与 eager 的耗时基本一致，说明 prefill 仍由
eager 支配，graph 主要影响 decode。

#### 10.7.4 剩余验收项

- AIME26 端到端精度；
- 正式 acceptance rate、TTFT/TPOT、吞吐与显存峰值对比；
- `num_speculative_tokens>1` 的 graph 路径；
- 若需要真正 32/64 同时 decode，需要降低每请求 KDA state/KV 占用或提高可用显存。

本轮日志：

- 4K 全 graph 并发：`/root/workspace/log/mtp_full_graph_4k_semantic_c64_20261113.log`
- 66560 全 graph 长上下文：`/root/workspace/log/mtp_full_graph_66560_semantic_20261113.log`
- 66560 MTP eager 长上下文：`/root/workspace/log/mtp_eager_66560_20261113.log`
- 并发语义结果：`work-tests/mtp/results/mtp_graph_20261113/concurrency_semantic_mtp_graph_4k_rounds2.json`
- 长上下文语义结果：`work-tests/mtp/results/mtp_graph_20261113/long_context_semantic_mtp_graph.json`

### 10.8 MTP + prefix cache 联调记录（2026-11-16）

#### 10.8.1 启动兼容性修复

当前 vLLM 已移除旧版 `mamba_utils.postprocess_mamba`，而 DLC V1 runner 仍在
`dlc_model_runner.py::_update_states_after_model_execute()` 中调用该入口。MTP +
prefix cache 启动时，8 个 worker 均在第一次 decode 前报：

```text
AttributeError: module 'vllm.v1.worker.mamba_utils' has no attribute 'postprocess_mamba'
```

已在 `vllm-cl/vllm_cl/patch/patch_mamba_state_copy.py` 增加
`_postprocess_mamba_compat()`：

- 保留旧版 align-mode postprocess 的 per-request bookkeeping；
- 复用 DLC 的 eager `_collect_mamba_copy_meta()` / `_do_mamba_copy_block()`，
  避免 Triton `batch_memcpy_kernel`；
- 仅在新版 `mamba_utils` 没有 `postprocess_mamba` 时安装兼容入口。

修复后服务可以正常启动，MTP eager 和 MTP graph 均可接受请求。该兼容层目前仍是
DLC 专用 fallback，尚未补 batch、多个 accepted token 和 CoW 的独立单测。

#### 10.8.2 冷/热 prefix cache 结果

测试条件：TP=8、`max-model-len=8192`、`gpu-memory-utilization=0.85`、
`block-size=256`、`--enable-prefix-caching`、`--mamba-cache-mode align`、
`--enable-mamba-shared-prefix-checkpoint`、`--no-async-scheduling`、MTP
`num_speculative_tokens=1`。固定 4096-token 随机 prompt，greedy 输出 32 token，
同一服务内连续发送 5 次。

| 配置 | 首次 / 第二次缓存 | 第 3 次起缓存 | greedy 输出是否冷热一致 |
|---|---:|---:|---|
| MTP graph，`max-num-batched-tokens=4096` | 0 / 0 | 3584 | **否** |
| MTP eager，`max-num-batched-tokens=4096` | 0 / 0 | 3584 | **否** |
| 无 MTP eager，`max-num-batched-tokens=4096` | 0 | 3840 | **否** |
| MTP graph，`max-num-batched-tokens=512` | 0 | 3584 | **是** |
| MTP eager，`max-num-batched-tokens=512` | 0 | 3584 | **是** |

4096-token chunk 配置中，冷请求输出为：

```text
197, 197, 322, 197, 197, 322, ...
```

第 3 次开始命中 3584 token 后输出变为：

```text
197, 197, 322, 13, 455, 6890, 198, ...
```

同样的输出翻转在无 MTP 的 eager 控制组也能复现，因此不是 MTP 独有，而是
KDA checkpoint/prefix-cache 通用问题。512-token chunk 下，MTP eager 和 graph
的冷、热请求输出均完全一致；缓存从第 2 次请求开始命中 3584 token。

#### 10.8.3 原因与暂定处理

4096-token prompt 在 `max_num_batched_tokens=4096` 时，checkpoint `3584` 位于
最后一次 prefill chunk 内部，当前 framework fallback 会在该位置把 prefill 拆成：

```text
[0, 3584] -> checkpoint state
[3584, 4096] -> final state
```

而冷请求执行的是完整 `[0, 4096]` prefill。两段 split 的 KDA kernel shape、FP32
归约顺序和 state staging 路径都与完整 prefill 不同，因此不保证 checkpoint state
与完整 prefill 的中间 state 逐 bit 一致；本次固定 prompt 恰好把微小差异放大成
greedy token 翻转。`--enable-mamba-shared-prefix-checkpoint` 只负责注册 shared-prefix
junction checkpoint，不能消除这个数值差异。

512-token chunk 使 3584 成为 chunk 末端，避开了非末端 checkpoint 的 split
fallback，因此冷热输出一致。该配置目前只作为打通 prefix-cache 路径的验证
workaround，不应直接作为生产配置：它把长上下文拆成更多 prefill step，性能尚未
在 16K/32K/64K 上实测。

该问题暂不要求在 framework 层继续用数值修补掩盖，待 KDA prefill checkpoint
原生算子落地后，以“checkpoint state 对完整 prefill snapshot 逐 bit 或 1 FP32 ULP
一致，且 cold/hot greedy token 完全一致”作为验收条件重新验证。算子 issue 中
当前写的 `<=2e-2` 只适合作为粗粒度数值门槛，不能单独证明 greedy 输出等价。

#### 10.8.4 当前剩余问题

1. **4096 chunk 非末端 checkpoint 仍会导致冷热输出翻转**：等待 KDA checkpoint
   原生算子；512 chunk 只是临时验证路径。
2. **512 chunk 的性能代价未测**：4096 档 cold 约 12.7 s、hot 约 3.4 s；16K/32K/64K
   尚需测量 chunk 数增加后的 prefill/TTFT，特别是 KPool select/top-k 的长上下文成本。
3. **4096 chunk + checkpoint flag 的 MTP graph capture 曾卡在 `0/1`**：
   日志停在 `Capturing CUDA graphs (decode, FULL): 0/1`，随后出现
   `No available shared memory broadcast block found` 60 s 告警；512 chunk graph
   可正常 capture。需要单独做 checkpoint flag、graph cache 和硬件残留状态 A/B，
   当前不能把 4096 graph 视为已打通。
4. **第二次相同请求仍可能不命中**：4096 和 eager 测试中前两次均为 0，第三次才命中；
   512 测试从第二次开始命中。需要确认是 request 完成后的 cache finalize 时序、
   EAGLE/MTP lookahead block drop，还是 metrics 读取窗口问题。
5. **缺少 512 chunk 的 no-prefix 基线**：目前只证明了 512 下 cold 与 hot 一致，
   尚未证明它与同 chunk 配置的 no-prefix cold 输出一致。
6. **多请求共享前缀未验证**：尚未覆盖多个请求共享 3584-token prefix、不同 suffix，
   以及每个请求分别与 cold baseline 对照。
7. **MTP draft cache/KPool 的 prefix reuse 未单独验证**：目标模型 KDA checkpoint
   之外，MTP drafter 还有独立的 KV cache group；需确认其 prefix 命中、lookahead
   block drop 和 KPool tail cache 的生命周期。
8. **async scheduling 未进入本轮**：当前统一使用 `--no-async-scheduling`；
   prefix-cache 路径打通后再验证 async scheduling 与 MTP graph 的组合。

本轮日志与结果：

- `/root/workspace/log/mtp_prefix_cache_ckpt_20261116.log`
- `/root/workspace/log/mtp_prefix_cache_ckpt_eager_20261116.log`
- `/root/workspace/log/prefix_cache_checkpoint_nomtp_eager_20261116.log`
- `/root/workspace/log/mtp_prefix_cache_ckpt_512_20261116.log`
- `/root/workspace/log/mtp_prefix_cache_ckpt_512_graph_20261116.log`
- `work-tests/mtp/results/prefix_cache_20261116/`

### 10.9 2026-11-16 复核：复位、命中时序与长上下文

#### 10.9.1 4096 graph 复位复核

先停止全部 vLLM 进程，执行：

```bash
cltechpd_clnt -s --full
```

随后以 4096-token chunk、MTP `num_speculative_tokens=1`、prefix cache 和
`FULL_DECODE_ONLY` graph 启动，8 个 worker 均完成 graph capture，服务正常进入
`Application startup complete`。上次停在 `Capturing CUDA graphs (decode, FULL): 0/1`
的现象本次未复现，判断为硬件/驱动残留状态导致，不是 4096 graph 的稳定代码缺陷。

但 `max-model-len=66560 + max-num-batched-tokens=512` 的长上下文 graph 仍有独立问题：

- `max-num-seqs=1`：capture 前停在 graph memory profiling，芯片现场在
  `custom_allreduce_2x4tpu4root_simple_bf16_rdma`；
- `max-num-seqs=8`：graph warmup 的 `_dummy_run()` 报
  `synErrorLaunchFailure`，芯片现场为 `custom_gather_tensor`
  的 `bad_vld_base: Vld base out of range`。

因此当前可把 4096 graph 视为可信启动路径，但不能把 512-token chunk 的长上下文
graph 视为已打通。本轮日志：

- 4096 graph 复位复核：`/root/workspace/log/mtp_prefix_cache_ckpt_after_reset_20261116.log`
- 4096 graph 精确 4096 请求：`/root/workspace/log/mtp_prefix_4096_graph_exact_20261116.log`
- 512 长上下文 graph 失败：`/root/workspace/log/mtp_prefix_512_longctx_graph_seq8_20261116.log`

失败调用链已由 `py-spy` 和 kernel log 交叉确认：

```text
GDNAttentionMetadataBuilder.build_for_cudagraph_capture
  -> GDNAttentionMetadataBuilder.build          # gdn_attn.py:230
  -> mamba_get_block_table_tensor                # utils.py:1136
  -> torch.gather(block_table, 1, indices)       # utils.py:1173
  -> gather_dlc_kernel                          # ScatterGatherKernel.cxx
  -> custom_gather_tensor                       # int32 input
```

`py-spy` 的 Python 栈停在 `gdn_attn.py:339` 的
`block_table_tensor[spec_sequence_masks_cpu, : self.num_spec + 1]`，这是因为
`custom_gather_tensor` 是异步 launch，报错可能在后续 Python 行才被观察到；kernel log
显示真正命中 `custom_gather_tensor` 的是前一步 `mamba_get_block_table_tensor()` 的
`torch.gather`。261 来自 `66560 / 256 + 1` 的 raw KDA block table 宽度，不是 batch
size；helper 输出才是 `[1,2]`。

instrumented 运行中，`gdn_attn.py:339` 实际拿到：

```text
block_table_tensor: shape=(1,2) stride=(2,1) storage_offset=0
                    ptr_mod512=0 int32 values=[[0,0]]
spec_sequence_masks_cpu: shape=(1,) cpu [True]
query_start_loc_cpu=[0,2], num_spec=1, num_spec_decodes=1
```

同轮 kernel log 中 `custom_gather_tensor` 已完成（cycles=4837），进程随后执行到
`custom_index_tensor_out_bf16` 并停在 graph capture；这次没有在该点复现
`bad_vld_base`，说明问题具有异步/时序或完整 graph 状态相关成分，不是稳定的
单 kernel 输入错误。

最小复现脚本已放在 `work-tests/mtp/repro_custom_gather.py`；脚本覆盖：

- `block_table[mask, :2]` 的原始 advanced indexing；
- `torch.gather(block_table, dim=0, index=[1,2])`；
- `torch.gather(block_table, dim=1, index=[1,2])` 的 helper 路径。

复位后实际执行结果：

| 测试 | 设备 | 次数 | 结果 |
|---|---|---:|---|
| eager `gather` / mask 三种路径 | dlc:0/1/6 单卡 | 1 | 全部通过 |
| eager 循环 | 8 卡并发 | 1,000/卡 | 全部通过 |
| graph capture + replay | dlc:1/6 单卡 | 100 | 全部通过 |
| graph capture + replay | 8 卡并发 | 1,000/卡 | 全部通过 |

单独的形状、dtype、mask/切片路径和循环 launch 都不能复现 `bad_vld_base`，说明
问题不只是一个简单的 `[1,261] int32 + [1,2] int64` gather。instrumented 运行也没在
该 gather 处重现，而是继续到后面的 `custom_index_tensor_out_bf16`；因此暂不能把
根因归到单一 gather API，下一步应保留这份 graph-capture 现场，重点检查 gather
之前的 kernel/VMEM 状态以及 8 rank 的异步推进差异。

补充 `max-num-seqs=8` 的 instrumented 结果：

```text
raw block table: shape=(8, 261), stride=(261, 1), offset=0, 全 0
helper output:   shape=(8, 2),   stride=(2, 1),   offset=0, 全 0
```

该轮在加入同步日志后正常到达 `Application startup complete`，没有出现
`custom_gather_tensor` 的 `bad_vld_base`。结合以下事实：

- 同一形状的 8 卡独立脚本 1000 次 eager/graph capture 全通过；
- `max-num-seqs=1` instrumented 轮次中 gather 也完成，随后卡在
  `custom_index_tensor_out_bf16`；
- `max-num-seqs=8` instrumented 轮次直接启动成功；

当前更倾向是完整模型 graph capture 的时序/竞态问题（日志同步改变了 launch 顺序），
而不是 block table 数值或单个 gather 形状错误。下一步应比较“无日志跑、日志跑、
单 rank 跑”的 kernel 时序，重点看 `custom_gather_tensor` 之前是否有未完成的
VMEM/CMEM producer kernel 或 rank 间 collective 等待。

#### 10.9.2 精确 4096 token 的命中时序

此前记录的“前两次 0、第三次才命中”在精确 4096-token prompt 上稳定复现：

| 配置 | 第 1 次 | 第 2 次 | 第 3 次 |
|---|---:|---:|---:|
| MTP graph，chunk=4096 | 0 hit，12.363 s | 0 hit，12.446 s | 3584 hit，3.536 s |
| MTP eager，chunk=512 | 0 hit，12.988 s | 3584 hit，3.768 s | 3584 hit，3.761 s |

4096 配置的 scheduler 日志显示第一次请求实际拆成 `3840 + 256`，第二次为
`3584 + 256 + 256`，第三次只重算最后 `256 + 256`。代码级根因在
`vllm/models/glm5next/common/kda.py::get_kv_cache_spec()`：

```python
if (
    current_platform.device_type != "dlc"
    or vllm_config.cache_config.mamba_cache_mode != "align"
    or vllm_config.speculative_config is not None
):
    return spec
```

当前实现只在**无 speculative config**时给 KDA spec 增加
`num_prefill_checkpoint_blocks=1` / `prefill_checkpoint_alignment=64`。MTP 打开后
该分支直接返回原始 spec，因此第一次 4096 请求不会生成 3584 的 Mamba checkpoint；
`--enable-mamba-shared-prefix-checkpoint` 此时只能等第二次请求从 full-attention
的 3584 前缀发现 shared-prefix junction，再补写并保留 Mamba state，第三次才形成
完整命中。这不是 prefix cache 的随机行为，而是 MTP 下 KDA checkpoint 尚未落地的
已知缺口。

用 `--prefix-cache-retention-interval 256` 复核后，前两次仍为 0、第三次才命中；
临时日志显示 Mamba manager 拿到的 checkpoint 为空，说明该参数不能替代 KDA
checkpoint 本身。因此不把它作为修复项，继续等原生 KDA checkpoint 算子。

时序相关结果文件：

- `work-tests/mtp/results/prefix_cache_20261116/mtp_prefix_4096_graph_exact_hit_timing_20261116.json`
- `work-tests/mtp/results/prefix_cache_20261116/mtp_prefix_4096_retention256_20261116.json`
- `work-tests/mtp/results/prefix_cache_20261116/mtp_prefix_512_exact4096_hit_timing_20261116.json`

#### 10.9.3 512 chunk 的关键对照

在 512-token chunk、MTP graph、prefix cache 开启的服务上，使用：

- 4000-token 固定 prompt `P`；
- `prefix=P[:3584]`，两个不同 4-token suffix `A`、`B`；
- 与同 chunk、`--no-enable-prefix-caching --mamba-cache-mode none` 的服务逐项对比。

结果：

| 请求 | prefix-on 输出 | prefix-off 输出 | prefix-on 是否一致 |
|---|---|---|---|
| `P`（4000 token） | 与 prefix-off 相同 | 基准 | 是 |
| `P[:3584]+A` 首次 | 与 prefix-off 相同 | 基准 | 是 |
| `P[:3584]+A` 重复 | 与首次相同 | 基准 | 是 |
| `P[:3584]+B` | 与 prefix-off 相同 | 基准 | 是 |

prefix-on 的共享前缀请求第一次命中 3072 token，重复请求和同前缀不同 suffix 的
`B` 均命中 3328 token；输出没有因命中位置不同而变化。说明 512 chunk 下
prefix-cache 的数值路径与 no-prefix 基线一致。结果文件：

- `work-tests/mtp/results/prefix_cache_20261116/mtp_prefix_512_on_20261116.json`
- `work-tests/mtp/results/prefix_cache_20261116/mtp_prefix_512_shared_prefix_20261116.json`
- `work-tests/mtp/results/prefix_cache_20261116/mtp_prefix_512_off_20261116.json`

#### 10.9.4 512 chunk 长上下文性能（eager）

长上下文 graph 启动未打通，先用同配置 eager 测 prefill 成本。配置为
`max-model-len=66560`、`max-num-batched-tokens=512`、MTP graph 关闭、
prefix cache 开启；每档发送同一 prompt 的冷/热两次，输出 4 token，记录流式首 token
时间（TTFT）和总耗时：

| prompt tokens | cold TTFT | cold total | hot TTFT | hot total | hot cache hit |
|---:|---:|---:|---:|---:|---:|
| 16,384 | 44.639 s | 45.001 s | 1.659 s | 2.002 s | 15,872 |
| 32,768 | 93.884 s | 94.087 s | 1.847 s | 2.050 s | 32,256 |
| 65,536 | 202.453 s | 202.638 s | 2.221 s | 2.417 s | 65,024 |

冷请求耗时随长度近似线性；热请求只重算尾部 512 token，TTFT 约 1.7–2.2 s。该表是
512 chunk workaround 的 prefill 量级数据，不是正式 SLA，也不包含长上下文 graph
decode 收益。结果文件：

- `work-tests/mtp/results/prefix_cache_20261116/mtp_prefix_512_longctx_eager_20261116.json`

#### 10.9.5 当前处理结论

1. 4096 graph 卡住已按硬件复位关闭；4096 graph 精确请求仍保留“两次预热、第三次
   命中 3584”的行为，等待 KDA checkpoint 算子消除。
2. 512 chunk 的 prefix-on/off 和共享前缀关键对照通过，单请求输出一致。
3. 命中时序已定位到 MTP 下 KDA spec 不启用 `num_prefill_checkpoint_blocks`，
   以及 shared-prefix junction 需要 sibling 请求才能发现。
4. 512 chunk 长上下文 eager 性能已测；512 chunk 长上下文 graph 仍有两个独立
   启动问题，需后续单独定位。

2026-11-17 补充：已合并并验证 KDA chunk-prefill checkpoint 远端实现
（kernel `7894c5129`、pytorch `ec181b5ff5`）。`dlc_kda_chunk_prefill` 的 checkpoint
功能通过静态/动态/syntest 和 4K/64K 补充验证；但
`dlc_kda_causal_conv_prefill` 仍没有 checkpoint API，完整 KDA checkpoint issue
尚未全部满足。详细报告见
`work-docs/glm53flash/KDA_prefill_checkpoint_operator_validation_2026-11-17.md`。

