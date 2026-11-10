# Hybrid MTP + Suffix Speculative Decoding（Model Runner V1）

## 1. 功能目标

`hybrid_mtp_suffix` 在 DLC Model Runner V1 中同时运行两路 proposer：

- MTP：继续使用当前 DLC 优化后的 padded batch、device metadata、KV cache 和 proposer graph。
- Suffix decoding：在 CPU bookkeeping 完成后，根据当前请求已生成的 token 查找重复后缀。

两路各自产生一条候选，但每个 request 每轮只选择一个 winner，并将 winner
送入现有 target verifier。实现不修改 rejection sampler，也不做双路联合验证。

```text
Target forward + sampling
        |
        +--> DLC MTP proposer ------> MTP candidate -----+
        |                                                |
        +--> CPU bookkeeping --> Suffix candidate -------+--> selector
                                                               |
                                                               v
                                                        single winner
                                                               |
                                                               v
                                                     existing verifier
```

## 2. 启动配置

当前仅支持 DLC Model Runner V1。对于会默认选择 V2 的模型，需要显式关闭 V2：

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
```

若 hybrid 配置最终仍选择了 Model Runner V2，启动会直接报错，不会静默退化为纯 MTP。

必要的 speculative 配置为：

```bash
--speculative-config '{
  "method": "hybrid_mtp_suffix",
  "num_speculative_tokens": 3
}'
```

MTP 默认复用 target model，不需要额外指定 draft model 路径。完整可选配置及默认值：

```json
{
  "method": "hybrid_mtp_suffix",
  "num_speculative_tokens": 3,
  "hybrid_preferred_method": "suffix",
  "hybrid_fallback": "mtp",
  "hybrid_min_history": 4,
  "hybrid_probe_interval": 16,
  "hybrid_hot_mtp_probe_interval": 16
}
```

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `hybrid_preferred_method` | `suffix` | 两路均非空但任一路历史不足时直接选择的方法；可设为 `mtp` |
| `hybrid_fallback` | `mtp` | 历史充分后，两路预计收益和长度都相同时的选择 |
| `hybrid_min_history` | `4` | 每种方法达到多少次 request-local 真实验证后，开始比较预计收益 |
| `hybrid_probe_interval` | `16` | 每个 request 每多少个 eligible round 强制探测一次 suffix；`0` 表示关闭 |
| `hybrid_hot_mtp_probe_interval` | `16` | suffix hot phase 中每多少轮恢复一次完整 MTP proposal；其余轮仅做 MTP first-pass KV maintenance |

Hybrid 会强制关闭 async scheduling。当前也不支持
`num_speculative_tokens_per_batch_size` 动态步数配置，因为 suffix proposer 要求固定的
`num_speculative_tokens`。

会影响选择结果的配置均纳入 `SpeculativeConfig.compute_hash()`。

## 3. Selector 决策顺序

Selector 为每个 request 独立维护历史。决策按以下优先级执行：

1. 两路都为空：本轮不发布 speculative token。
2. 只有一路非空：选择唯一可用的方法。
3. 请求处于 suffix hot phase：只要 suffix 非空就选 suffix。
4. MTP 已有至少 2 次观察、平滑 acceptance 低于 50%，且 suffix 不足 2 次观察：立即选择 suffix 收集样本。
5. 到达 `hybrid_probe_interval`：选择 suffix 做周期探测。
6. 任一路观察数不足 `hybrid_min_history`：选择 `hybrid_preferred_method`。
7. 两路历史均充分：比较预计接受 token 数。
8. 预计收益相等时比较候选长度；仍相等则使用 `hybrid_fallback`。

预计收益为：

```text
expected_accepted_tokens = candidate_length * smoothed_acceptance
smoothed_acceptance = (accepted_tokens + 1) / (drafted_tokens + 2)
```

只有真正送入 verifier 的 winner 才更新 acceptance。Loser 没有经过验证，不能记为
rejection。

## 4. Suffix hot phase

Suffix 的优势通常具有明显阶段性：进入重复输出区域后会连续高命中。因此实现了带迟滞的
request-local hot phase：

- 连续 2 次 suffix proposal acceptance `>= 80%`：进入 hot phase。
- Hot phase 中只要 suffix 非空，就持续选择 suffix。
- 单次低命中不会退出，避免在边界处频繁振荡。
- 连续 2 次 suffix proposal acceptance `< 50%`：退出 hot phase。
- Suffix 某轮没有候选时回退 MTP，但不会仅因此清除 hot 状态。

日志中的 `suffix_hot=1` 表示当前请求处于该阶段。

## 5. MTP KV cache 连续性

MTP proposer 始终由 `DLCModelRunner` 直接持有，没有包进新的 proposer wrapper，因此继续
命中 `patch_eagle.py` 中基于真实 proposer 实例的 DLC graph、metadata 和 KV cache 路径。

普通阶段中，即使某轮选择 suffix，MTP candidate 仍会完整生成。Suffix hot phase 中执行
first-pass-only MTP maintenance：仍吸收 target 最终确认的 token 并写入 KV，但跳过后续 serial
draft steps。默认每 16 个 hot round 恢复一次完整 MTP proposal 作为 probe。

当前 V1 proposer 的步数由 `propose(num_speculative_tokens=...)` 显式传入；移植实现会在
maintenance round 同时把该参数和 proposer 临时状态设为 `1`，确保 first-pass-only 行为不会被
当前 API 覆盖。

下一轮继续通过原生 rejected-token 修正和 first-pass replay 同步 MTP 状态，包括：

- `prepare_next_token_ids_padded()`
- `prepare_inputs_padded()`
- `num_rejected_tokens_gpu`
- sequence length、position 和 slot mapping 修正

因此 suffix winner 不会切断 MTP KV cache。

## 6. 日志

默认仅 tensor-parallel rank 0 每 32 个 proposal batch 输出一次聚合日志：

```text
Hybrid MTP+suffix: winners={'mtp': 122, 'suffix': 158}
sample_req=req-1 mtp_mode=maintenance mtp_len=1 suffix_len=3 winner=suffix
request_stats={
  'mtp_acceptance': 0.62,
  'suffix_acceptance': 0.99,
  'mtp_observations': 61,
  'suffix_observations': 65,
  'suffix_hot': 1
}
```

`winners` 是 worker 启动以来的累计选择次数；`*_observations` 只统计已经获得 verifier
反馈的 winner，因此可能比 winner 计数少一轮。

日志频率通过环境变量控制：

```bash
# 每轮输出，适合功能调试
export DLC_HYBRID_MTP_SUFFIX_LOG_INTERVAL=1

# 每 10 个 proposal batch 输出
export DLC_HYBRID_MTP_SUFFIX_LOG_INTERVAL=10

# 完全关闭
export DLC_HYBRID_MTP_SUFFIX_LOG_INTERVAL=0
```

## 7. 推荐验证方式

保持模型、prompt、sampling 参数和并发一致，至少比较以下四组：

1. 无 speculative decoding
2. 纯 MTP
3. 纯 suffix
4. `hybrid_mtp_suffix`

建议记录端到端耗时、generation throughput、TPOT、Mean acceptance length、
Accepted/Drafted throughput、per-position acceptance、winner 分布、suffix hot phase
进入/退出位置，以及 MTP DLC graph capture/replay 是否命中。

Suffix 优势测试应使用 token 序列高度重复的输出，并使用 `temperature=0` 和足够大的
`max_tokens`。例如要求模型将同一句话逐行重复数百次。

## 8. 当前限制

- 仅接入 DLC Model Runner V1；Model Runner V2 会在配置阶段明确拒绝。
- 两路不是 target 级联合验证，只能选择一个 winner。
- Suffix loser 和 MTP loser 都没有真实 acceptance，统计只来自 winner。
- Hot phase 的 `80%/50%` 阈值和连续次数目前是固定常量。
- Hot phase 仍运行 MTP first pass 维护 KV，只跳过后续 serial draft steps。
- 当前移植覆盖 selector/config 静态回归；DLC 真机正确性和性能必须以实际服务 A/B 为准。
