# GLM-5.3-Flash MTP 端到端排查记录

日期：2026-11-10  
范围：`release-glm53flash` + TYD 真机 + eager MTP（`num_speculative_tokens=1`）  
结论：功能链路已打通到“可完成 MTP 请求”，但 greedy 输出仍存在间歇性错误，
当前不能判定 MTP 正确性验收通过。

## 1. 当前结论

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

