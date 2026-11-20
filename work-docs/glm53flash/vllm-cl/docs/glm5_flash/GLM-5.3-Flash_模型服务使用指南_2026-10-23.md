# GLM-5.3-Flash 模型服务使用指南（2026-10-23）

> 2026-11-13 更新：新增 MTP + `FULL_DECODE_ONLY` graph 的验证/灰度启动说明（3.6）。
> 2026-11-18 更新：补充 `--enable-prompt-tokens-details`，用于观测 prefix cache 的
> `usage.prompt_tokens_details.cached_tokens`。

## 1. 交付内容

本指南适用于在 8 卡 TYD 环境上部署 GLM-5.3-Flash-Mix 推理服务。部署直接使用已安装的 `vllm` 命令行，不依赖 `work-docs` 中的启动脚本。

生产配置默认开启 `FULL_DECODE_ONLY` graph 捕获，prefill 和 chunked prefill 保持 eager。Graph 对单请求和中等并发 decode 有明显收益；32K/更高并发的性能仍受请求形状和 graph bucket 影响。

MTP + `FULL_DECODE_ONLY` graph 的验证/灰度启动方式见 3.6。当前生产默认仍是不启用
MTP；在 AIME26 和正式 acceptance/性能验收完成前，不要把 3.6 当作默认生产配置。

## 2. 环境前提

| 项目     | 要求                                                                            |
| -------- | ------------------------------------------------------------------------------- |
| 加速卡   | 8 × TYD，`/dev/cltech0..7` 可见                                              |
| 模型目录 | 默认 `/data/models/GLM-5.3-Flash-Mix`                                         |
| vLLM     | 已安装到 `/root/workspace/.venv`，提供 `/root/workspace/.venv/bin/vllm`     |
| vLLM-CL  | 已安装到同一 Python 环境，提供 `dlc` 与 `dlc_patches` 插件                  |
| Python   | `/root/workspace/.venv/bin/python`，Python 3.10                               |
| 驱动     | `Driver Version_v26.0.1`，8 卡均为 `Api_Version_26`                         |
| 媒体依赖 | `torchaudio 2.5.0+dlc`、`torchvision 0.19.0+dlc`，与 DLC PyTorch 2.5.0 匹配 |

启动前检查：

```bash
ls -l /dev/cltech{0..7}
ls -ld /data/models/GLM-5.3-Flash-Mix
/root/workspace/.venv/bin/python -c "import torch, vllm, vllm_cl; print(torch.__version__)"
```

如果服务曾经异常退出或硬件初始化卡住，先确认没有残留进程，再重置硬件：

```bash
pkill -f 'vllm serve /data/models/GLM-5.3-Flash-Mix' || true
cltechpd_clnt -s --full
```

## 3. 推荐启动方式

### 3.1 已验证配置

以下配置对应本次发布 smoke，建议作为首次部署和验收基线。先导出运行环境变量：

```bash
export PATH=/root/workspace/.venv/bin:$PATH
export VLLM_PLUGINS=dlc,dlc_patches
export DLC_SKIP_VERSION_CHECK=0
export DLC_SYN_GRAPH_CACHE=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_USE_DLC_COL_MAJOR_MATMUL=1
export PYTORCH_DLC_ALLOC_CONF=expandable_segments:True
```

直接启动服务：

```bash
vllm serve /data/models/GLM-5.3-Flash-Mix \
  --served-model-name GLM-5.3-Flash-Mix \
  --language-model-only \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 64 \
  --block-size 256 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.80 \
  --trust-remote-code \
  --port 18005 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kernel-config '{"enable_jit_warmup": false}'
```

如需验证 prefix cache 命中并读取每次请求的缓存 token 数，在启动命令中追加
`--enable-prompt-tokens-details`。该开关只增加响应中的统计字段，不改变缓存或计算
路径；使用仓库启动脚本时可写 `--prompt-token-details`。具体响应见 5.6。

### 3.2 参数说明

| 参数                         | 取值                                      | 用法与注意事项                                                                                                                                                          |
| ---------------------------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 模型目录（位置参数）         | `/data/models/GLM-5.3-Flash-Mix`        | `vllm serve` 后的第一个位置参数，指向本地模型目录；目录必须包含完整权重、config 和 tokenizer。                                                                        |
| `--served-model-name`      | `GLM-5.3-Flash-Mix`                     | 对外 API 的模型名。客户端请求体中的 `model` 必须与此值一致。                                                                                                          |
| `--language-model-only`    | 开关                                      | 只加载语言模型路径，关闭多模态输入；GLM 文本推理应保持开启。                                                                                                            |
| `--dtype`                  | `bfloat16`                              | 模型权重与激活数据类型。当前 checkpoint 和 DLC kernel 按 BF16 验证，不要改成 `auto` 或 FP16。                                                                         |
| `--max-model-len`          | `4096`                                  | 单请求最大上下文，包含 prompt 和输出 token；同时决定 KV cache 上限。首次部署先用 4096，按剩余 KV 容量逐步提高。8 卡 TYD 在 0.80 显存利用率下理论上限约 171K，详见 3.5。 |
| `--max-num-batched-tokens` | `4096`                                  | 单次调度最多处理的 token 数，也是 chunked prefill 的单块预算；应与 `--max-model-len` 一起调整。                                                                       |
| `--max-num-seqs`           | `64`                                    | 单个 iteration 最多并发处理的序列数。吞吐与 decode graph bucket 相关，需按压测结果调整。                                                                                |
| `--block-size`             | `256`                                   | KV cache 页大小。当前 DLC 稀疏 MLA/KDA 布局按 256 验证，生产环境不要修改。                                                                                              |
| `--tensor-parallel-size`   | `8`                                     | 单机 8 卡张量并行度，应与可见 TYD 卡数一致。                                                                                                                            |
| `--gpu-memory-utilization` | `0.80`                                  | 每卡可供模型权重、KV cache 和运行时使用的内存比例。提高前先确认剩余容量；过高会导致 KV cache 或图捕获 OOM。                                                             |
| `--trust-remote-code`      | 开关                                      | 允许加载 checkpoint 自带模型代码；当前模型必须开启。                                                                                                                    |
| `--port`                   | `18005`                                 | OpenAI 兼容 API 监听端口。未指定 `--host` 时监听所有地址。                                                                                                            |
| `--compilation-config`     | `{"cudagraph_mode":"FULL_DECODE_ONLY"}` | 只捕获 uniform decode graph，prefill 和 chunked prefill 保持 eager；不要切换为完整 graph。                                                                              |
| `--kernel-config`          | `{"enable_jit_warmup": false}`          | 关闭 JIT warmup，使用发布时已编译并验证的 AOT kernel。                                                                                                                  |
| `--enforce-eager`          | 开关                                      | 关闭 graph capture，所有阶段走 eager；仅用于精度排查、问题复现或性能对照，不作为生产默认值。                                                                            |
| `--enable-prompt-tokens-details` | 开关                                 | 在 OpenAI completion/chat 响应的 `usage.prompt_tokens_details.cached_tokens` 中返回本次请求命中的前缀 token 数。默认关闭；只影响响应字段，不改变 prefix-cache 的启用或计算。启动脚本对应开关为 `--prompt-token-details`。 |

环境变量说明：

| 变量                              | 取值                         | 用法与注意事项                                                                                                                                                      |
| --------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `VLLM_PLUGINS`                  | `dlc,dlc_patches`          | 必须同时包含两项：`dlc` 启用 DLC 平台，`dlc_patches` 安装 GLM 的模型与 KV-cache patch。                                                                         |
| `DLC_SKIP_VERSION_CHECK`        | `0`                        | 严格校验驱动、Synapse 与 Runtime ABI；仅隔离的旧测试镜像可临时设为 `1`。                                                                                          |
| `DLC_SYN_GRAPH_CACHE`           | `1`                        | 开启 Synapse graph replay 的 kernel-batch 缓存。默认值为 `0`；它不控制 graph capture，但关闭时每次 replay 都重新遍历并下发图节点，损失生产环境的 graph 下发优化。 |
| `VLLM_USE_DLC_COL_MAJOR_MATMUL` | `1`                        | 使用适配 checkpoint 的 DLC column-major linear 路径。                                                                                                               |
| `PYTORCH_DLC_ALLOC_CONF`        | `expandable_segments:True` | 缓解 Mamba state 与 MLA tensor 交错分配造成的显存碎片。                                                                                                             |
| `VLLM_LOGGING_LEVEL`            | `INFO`                     | 避免开发 shell 的 DEBUG 日志放大；排障时可临时改为 `DEBUG`。                                                                                                      |
| `GLM5_PROMPT_TOKEN_DETAILS`     | `0`                        | 设为 `1` 时启动脚本自动追加 `--enable-prompt-tokens-details`；也可在命令行使用 `--prompt-token-details`。默认关闭。                                                |

冷启动约 3–4 分钟：权重加载约 110 s，KV/profile/graph 初始化约 43 s。启动期间不要重复拉起第二个实例。

### 3.3 后台启动

建议使用 tmux 或 systemd 管理服务。临时后台启动示例：

```bash
mkdir -p /root/workspace/log
setsid vllm serve /data/models/GLM-5.3-Flash-Mix \
  --served-model-name GLM-5.3-Flash-Mix \
  --language-model-only \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 64 \
  --block-size 256 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.80 \
  --trust-remote-code \
  --port 18005 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kernel-config '{"enable_jit_warmup": false}' \
  > /root/workspace/log/glm5_service.log 2>&1 &

SERVICE_PID=$!
echo "service pid: $SERVICE_PID"
```

健康检查：

```bash
until curl -fsS --max-time 5 http://127.0.0.1:18005/health >/dev/null; do
  sleep 5
done
echo ready
```

停止：

```bash
kill -- -$SERVICE_PID
```

如果 PID 已丢失，先定位再停止，不要直接杀无关进程：

```bash
pgrep -af 'vllm serve /data/models/GLM-5.3-Flash-Mix'
pkill -f 'vllm serve /data/models/GLM-5.3-Flash-Mix'
```

### 3.4 更大上下文

`--max-model-len` 越大，KV cache 占用越高。测试机在 4096 上下文、0.80 显存利用率下已验证；更长上下文必须先确认剩余 KV 容量，再用逐步增大的方式验证。以 8192 为例，完整命令如下：

```bash
vllm serve /data/models/GLM-5.3-Flash-Mix \
  --served-model-name GLM-5.3-Flash-Mix \
  --language-model-only \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 64 \
  --block-size 256 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --port 18005 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kernel-config '{"enable_jit_warmup": false}'
```

如果启动阶段报 KV cache 容量不足，降低 `--max-model-len` 或提高 `--gpu-memory-utilization`；不要在没有实测的情况下直接使用 128K/1M 配置。

### 3.5 上下文长度上限

模型 checkpoint 声明的 `max_position_embeddings` 是 **1,048,576**，但 8 卡 TYD
无法为这个长度分配 KV cache，实际 `--max-model-len` 上限由每张卡的 KV block
预算决定。当前 GLM 混合缓存的单请求 block 数可写为：

```text
blocks_per_request(L) = 69 + 2 × ceil(L / 256)
```

其中 69 来自 34 个 KDA 层各 2 个 state block，加上 1 个 KPool tail block；后面
两项分别对应 MLA 与 KPool 的 token 相关缓存。`blocks_per_request(L)` 必须不超过
启动时报告的 KV block 总数。

| `gpu_memory_utilization` |                            KV block 预算 |           `max-model-len` 理论上限 | 说明                                                   |
| -------------------------: | ---------------------------------------: | -----------------------------------: | ------------------------------------------------------ |
|                       0.80 |  1,411 blocks（实测启动日志，28.58 GiB） |           **171,776，约 171K** | 当前生产配置；达到该长度时并发能力为 1.00x，已无余量   |
|                       0.95 | 约 1,883–1,891 blocks（按显存线性外推） | **约 232K，最高不超过约 233K** | 仅理论估算；只剩 5% 非 KV 余量，风险高，不作为生产配置 |

0.95 不是推荐值。显存利用率提高后，模型权重、graph、分配器碎片和驱动开销仍占用
固定空间，实际可用 block 数可能低于线性估算；必须通过启动日志中的
`Available KV cache memory`、`GPU KV cache size` 和 `Maximum concurrency` 确认。

当前已验证的常规配置仍是 4096；`131072` 曾在 0.85 显存利用率、eager 路径下完成
长上下文验证。171K/232K 是容量上限，不代表长上下文的时延可接受：prefill 会随
token 数近似线性增长，达到上限时也只能同时服务一个满长请求。

### 3.6 MTP + FULL_DECODE_ONLY graph（验证/灰度）

当前已验证的组合是 `method=mtp`、`num_speculative_tokens=1`、TP8、
`FULL_DECODE_ONLY` decode graph。MTP draft 和 target verify 都使用 graph；
prefill/chunked prefill 仍是 eager。`num_speculative_tokens=1` 是当前验证值，
不是代码硬限制；`>1` 尚未验证。

环境变量在 3.1 节基础上保持不变；长上下文服务额外放宽执行超时：

```bash
export PATH=/root/workspace/.venv/bin:$PATH
export VLLM_PLUGINS=dlc,dlc_patches
export DLC_SKIP_VERSION_CHECK=0
export DLC_SYN_GRAPH_CACHE=1
export VLLM_USE_DLC_COL_MAJOR_MATMUL=1
export PYTORCH_DLC_ALLOC_CONF=expandable_segments:True
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
```

如果使用仓库的 `start_glm5_flash_vllm_cl.sh`，graph 默认已经开启，不需要额外传
`--graph`。4K 启动示例：

```bash
cd /root/workspace/work-docs/glm53flash/vllm-cl/docs/glm5_flash
PORT=18005 \
MAX_MODEL_LEN=4096 \
GPU_MEMORY_UTILIZATION=0.85 \
bash start_glm5_flash_vllm_cl.sh \
  --no-enable-prefix-caching \
  --mamba-cache-mode none \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 64 \
  --no-async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --kernel-config '{"enable_jit_warmup": false}'
```

长上下文把 `MAX_MODEL_LEN` 改成 `66560`、`--max-num-batched-tokens` 改成 `4096`
即可。启动脚本默认设置 `DLC_SYN_GRAPH_CACHE=1`、`VLLM_PLUGINS=dlc,dlc_patches`
和 `FULL_DECODE_ONLY`；显式传 `--eager` 才会关闭 graph。

**4K 验证/灰度配置（并发语义已验证）**

```bash
vllm serve /data/models/GLM-5.3-Flash-Mix \
  --served-model-name GLM-5.3-Flash-Mix \
  --language-model-only \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 64 \
  --block-size 256 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --port 18005 \
  --no-enable-prefix-caching \
  --mamba-cache-mode none \
  --no-async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kernel-config '{"enable_jit_warmup": false}'
```

**长上下文配置（已验证到 prompt 64,625 token + 32 token 输出）**

```bash
vllm serve /data/models/GLM-5.3-Flash-Mix \
  --served-model-name GLM-5.3-Flash-Mix \
  --language-model-only \
  --dtype bfloat16 \
  --max-model-len 66560 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 64 \
  --block-size 256 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --port 18005 \
  --no-enable-prefix-caching \
  --mamba-cache-mode none \
  --no-async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kernel-config '{"enable_jit_warmup": false}'
```

**为什么验证命令带这三个开关**

| 参数 | 当前用途 | 是否 MTP 必需 |
|---|---|---|
| `--no-enable-prefix-caching` | 避免 prefix cache 命中/未命中改变 KDA/KPool state 复用和 batch 组成，便于稳定复现 | 否 |
| `--mamba-cache-mode none` | 禁用 Mamba/KDA state 的跨请求复用；关闭 prefix caching 时框架本身也会自动设为 `none`，这里显式写出便于审计 | 否；显式冗余 |
| `--no-async-scheduling` | 当前 MTP ↔ graph ↔ KDA state 的验证全部在同步调度下完成，避免异步调度改变 metadata 更新顺序 | 否；尚未验证异步调度组合 |

因此这三项不是“MTP 必须这样写”，而是当前验证基线的保守配置。若要逐项打开，
建议一次只去掉一个，并重新跑健康检查、单请求、并发语义和长上下文；不要直接把
三项全部删除后当成已验证生产配置。

**`num_speculative_tokens` 是否只能为 1**

不是。代码路径按 `num_speculative_tokens` 动态计算 MTP proposer 步数、target
verify token 数、graph capture size、KDA conv state 宽度和 KPool transaction
范围，理论上支持大于 1。当前只验证了 1，原因是要先把 MTP + graph 的基础路径和
padding/回滚边界收敛。DLC KDA recurrent kernel 还有单请求 token-state page
上限（当前为 16），graph capture size、显存和验证开销也会随 draft 深度增长；
建议先把 2 作为下一档，不要直接跳到很大的值。提高到 2 或更大时需要重新验证：

- target verify 的 graph bucket 和 `max-num-batched-tokens` 是否覆盖扩大的 token 数；
- KDA conv/recurrent state 宽度、KPool 事务和拒绝回滚；
- draft acceptance、TTFT/TPOT 和显存峰值；
- 长上下文下的 marker 精度和稳定性。

启动日志应同时确认：

- `FULL_DECODE_ONLY`；
- `Profiling CUDA graph memory: FULL=...` 和 `Graph capturing finished`；
- 请求后出现 `SpecDecoding metrics`；
- 如需确认 proposer graph 真正 replay，可加：

```bash
export DLC_MTP_DRAFTER_CUDAGRAPH_DEBUG=1
export DLC_MTP_DRAFTER_CUDAGRAPH_DEBUG_FILE=/tmp/mtp_graph_replay.log
```

日志中应出现 `dlc_mtp_proposer_graph replay tokens=... reqs=... steps=1`。

容量与验收口径：

- MTP 增加第 45 层 draft 权重和 draft cache，验证配置使用 `0.85` 显存利用率；
  不要直接沿用无 MTP 的 `0.80` 作为硬性前提。
- 4K 配置下客户端 in-flight 1/8/16/32/64 语义请求均验证通过；但受 KDA state/KV
  容量限制，服务端实际同时 `Running` 峰值约 20，32/64 档是排队执行。不要把
  “64 in-flight”误解为“64 个请求同时 decode”。
- 长上下文已验证 17K/34K/64.6K prompt 的 marker 检索，MTP eager 与 graph 都通过。
- 单请求固定 greedy 已验证 20/20；AIME26、正式 acceptance rate、TTFT/TPOT 和
  吞吐/显存峰值仍需单独验收。
- 若 MTP graph 启动失败，先用同一 `--speculative-config` 加 `--enforce-eager`
  做对照。eager 正常而 graph 失败时，保留启动日志并检查是否为 KDA metadata、
  graph bucket padding 或 KPool state mapping 问题。

## 4. Graph 与 eager

### 4.1 默认 graph

生产命令中的 graph 参数：

```bash
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
--kernel-config '{"enable_jit_warmup": false}'
```

Graph 启动前还必须设置：

```bash
export DLC_SYN_GRAPH_CACHE=1
```

`--compilation-config` 负责启用 capture；`DLC_SYN_GRAPH_CACHE=1` 负责缓存首次 replay 记录的 kernel queue batch，后续 replay 直接复用缓存配置，减少逐节点 host 下发。未设置时 Synapse 按默认值 `0` 处理，graph 仍能运行，但不在生产性能路径上。

实际启动日志中应确认 `FULL_DECODE_ONLY`，并出现每个 Worker 的 graph 内存统计：

```text
Profiling CUDA graph memory: FULL=35 (largest=256)
```

- 35 张图覆盖最大 256 个 decode token。
- Graph 只覆盖 uniform decode；prefill/chunked prefill 不走 graph。
- 当前 DLC 构建把 `torch.compile` 替换为透传，因此 piecewise graph 没有可用子图。
- 当前环境自动使用 V1 model runner；V2 runner 要求 Triton，未在本环境启用。

### 4.2 eager 对照

只在排查精度、复现问题或比较输出时使用。删除上面生产命令中的 `--compilation-config` 和 `--kernel-config`，并追加 `--enforce-eager`：

```bash
vllm serve /data/models/GLM-5.3-Flash-Mix \
  --served-model-name GLM-5.3-Flash-Mix \
  --language-model-only \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 64 \
  --block-size 256 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.80 \
  --trust-remote-code \
  --port 18005 \
  --enforce-eager
```

## 5. API 调用

### 5.1 健康检查与模型列表

```bash
curl -sS http://127.0.0.1:18005/health
curl -sS http://127.0.0.1:18005/v1/models
```

服务名默认为 `GLM-5.3-Flash-Mix`。

### 5.2 非流式 Chat Completions

```bash
curl -sS --max-time 600 http://127.0.0.1:18005/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "GLM-5.3-Flash-Mix",
    "messages": [{"role": "user", "content": "请介绍一下你自己。"}],
    "max_tokens": 1024,
    "temperature": 0,
    "stream": false
  }'
```

### 5.3 流式输出

```bash
curl -N --max-time 600 http://127.0.0.1:18005/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "model": "GLM-5.3-Flash-Mix",
    "messages": [{"role": "user", "content": "写一段一百字左右的自我介绍。"}],
    "max_tokens": 256,
    "temperature": 0,
    "stream": true
  }'
```

### 5.4 OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18005/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="GLM-5.3-Flash-Mix",
    messages=[{"role": "user", "content": "用一句话说明芯片软件栈的作用。"}],
    temperature=0,
    max_tokens=128,
)
print(response.choices[0].message.content)
```

### 5.5 Token 数检查

长上下文请求先用 `/tokenize` 确认总 token 数，给输出预留空间：

```bash
curl -sS http://127.0.0.1:18005/tokenize \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"GLM-5.3-Flash-Mix","prompt":"待检查的文本"}'
```

Chat Completions 的 `prompt_tokens` 包含 chat template 增加的控制 token，通常比原始文本估算略大。

### 5.6 Prefix-cache 命中观测

`cached_tokens` 需要服务端显式开启 `--enable-prompt-tokens-details` 才会出现在
响应中。使用启动脚本时可以直接传：

```bash
bash start_glm5_flash_vllm_cl.sh \
  --eager \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prompt-token-details \
  ...
```

对应的直接 `vllm serve` 参数为：

```text
--enable-prompt-tokens-details
```

开启后，非流式 completion 的 `usage` 会形如：

```json
{
  "prompt_tokens": 4096,
  "completion_tokens": 32,
  "prompt_tokens_details": {
    "cached_tokens": 3584
  }
}
```

读取时注意：

- 第一次请求没有可复用前缀，`cached_tokens` 通常为 `0`，不是缓存故障。
- 后续相同前缀请求才可能返回非零值；实际值受 `--block-size 256`、KDA checkpoint
  边界和多组 KV cache 命中归约影响，可能小于 prompt 总长度。
- `cached_tokens` 只用于观测和验收；关闭该参数时字段为 `null` 或不出现，不代表
  prefix cache 未命中，仍应结合请求耗时和服务端 `Prefix cache hit rate` 判断。
- 该参数不控制 `--enable-prefix-caching`；两者需要分别配置。

## 6. 日志与诊断

### 6.1 服务日志

- 前台启动：日志直接打印到终端，建议使用 `tee` 留存。
- 后台启动：日志写入 `/root/workspace/log/glm5_service.log`。
- 日志中应重点确认：`FULL_DECODE_ONLY`、`FULL=35`、`Application startup complete`、KV cache 容量和 `Engine ...` 吞吐统计。

### 6.2 Profiling 与 kernel log

Profiling 需要同时设置环境变量并在 `vllm serve` 命令中追加 profiler 参数。在 3.1 节环境变量基础上，启动前再导出：

```bash
export VLLM_TORCH_PROFILER_TOKENS=5
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
export DLC_SYN_PROF_CYCLE=1
export DLC_SYN_PROF_DIR=/tmp/syn_log/syn_trace
export DLCCL_SYNC_ALL_MACHINE=1
mkdir -p /tmp/syn_log /tmp/syn_log/syn_trace

vllm serve /data/models/GLM-5.3-Flash-Mix \
  --served-model-name GLM-5.3-Flash-Mix \
  --language-model-only \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 64 \
  --block-size 256 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.80 \
  --trust-remote-code \
  --port 18005 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kernel-config '{"enable_jit_warmup": false}' \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/syn_log","torch_profiler_with_stack":true}'
```

`VLLM_TORCH_PROFILER_TOKENS=5` 表示每次 `/start_profile` 后记录 5 个模型 step；设为 `0` 表示持续记录到 `/stop_profile`。Trace 写入 `/tmp/syn_log`，可用 Perfetto 打开 `*.pt.trace.json.gz`。服务启动完成后，另开终端采集：

```bash
curl -sS -X POST http://127.0.0.1:18005/start_profile
curl -sS http://127.0.0.1:18005/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"GLM-5.3-Flash-Mix","messages":[{"role":"user","content":"ping"}],"max_tokens":2,"temperature":0}'
curl -sS -X POST http://127.0.0.1:18005/stop_profile
```

Kernel 级日志不增加 `vllm serve` 参数，而是启动前导出以下变量：

```bash
export DLC_SYN_LOG_DIR=/root/workspace/log/syn_log
export DLC_SYN_LOG_PREFIX=syn
export DLC_SYN_VERBOSE=1
export DLC_SYN_DEBUG=1
export DLC_SYN_PROF_CYCLE=1
export DLCCL_SYNC_ALL_MACHINE=1
mkdir -p "$DLC_SYN_LOG_DIR"

# 然后使用 3.1 节的 vllm serve 命令启动
```

该模式非常冗长，只用于短请求诊断。每个进程写一个 `<prefix>_<pid>.ansi` 到 `/root/workspace/log/syn_log`，包含每次 kernel launch 的形状、stride、dtype、标量参数和 device cycle。

### 6.3 芯片状态与卡住排查

如果请求长时间处于 running 或服务停止响应：

```bash
cd /usr/local/chipltech/cl_tools/CsrAccess
./peek.sh -s status -t 0       # 卡 0 状态
./peek_stuck.sh -t 0           # 卡 0 当前卡在哪个算子
```

需要重置硬件时，先停止所有占用芯片驱动的进程，再执行：

```bash
cltechpd_clnt -s --full
```

## 7. 已验证功能与性能基线

### 7.1 发布 smoke 结果

配置：TP=8、`max_model_len=4096`、`max_num_batched_tokens=4096`、`max_num_seqs=64`、`gpu_memory_utilization=0.80`、`block_size=256`、graph 默认开启。

| 场景                            | 结果                                |
| ------------------------------- | ----------------------------------- |
| 单请求 20 prompt / 32 output    | 2.107 s                             |
| 8 并发，每请求 20 / 32          | wall 3.751 s，约 68.24 output tok/s |
| 长上下文 3610 prompt / 8 output | 72.419 s                            |
| 健康检查                        | HTTP 200                            |
| Graph 捕获                      | 35 张 decode graph，最大 256 token  |

### 7.2 Graph 与 eager、graph replay cache 参考

| 场景                | 并发 | Graph gap p50 | Eager gap p50 | Graph tok/s | Eager tok/s |
| ------------------- | ---: | ------------: | ------------: | ----------: | ----------: |
| 49 prompt + 96 out  |    1 |      67.64 ms |     110.27 ms |       14.11 |        8.83 |
| 49 prompt + 96 out  |    8 |     103.49 ms |     112.86 ms |       65.38 |       59.95 |
| 49 prompt + 64 out  |   32 |     222.04 ms |     222.88 ms |      100.08 |      100.55 |
| 49 prompt + 64 out  |   64 |     277.89 ms |     263.59 ms |       92.10 |       99.12 |
| 860 prompt + 64 out |    1 |      73.34 ms |      96.03 ms |       11.27 |        8.99 |
| 860 prompt + 64 out |    8 |     148.58 ms |     144.84 ms |       30.46 |       31.22 |
| 1217 prompt + 8 out |    1 |  TTFT 2.502 s |  TTFT 2.498 s |          — |          — |

性能数字来自当前 8 卡测试机，只应作为部署参考，不应直接当作其他硬件批次或驱动版本的 SLA。

> 注：上述历史 Graph/eager 数据采集时未显式设置 `DLC_SYN_GRAPH_CACHE`，Synapse 按默认值 `0` 运行。生产配置使用 `DLC_SYN_GRAPH_CACHE=1`；下表为 2026-10-26 使用相同服务参数完成的生产配置复测。

生产推荐配置 `DLC_SYN_GRAPH_CACHE=1` 的单并发参考（每档重复 3 次取中位数）：

| 场景                | 并发 | cache=0 gap p50 | cache=1 gap p50 | cache=1 吞吐 | 相对 cache=0 变化                    |
| ------------------- | ---: | --------------: | --------------: | -----------: | ------------------------------------ |
| 49 prompt + 96 out  |    1 |        63.78 ms |        57.58 ms |  16.53 tok/s | gap -9.7%，吞吐 +10.9%               |
| 860 prompt + 64 out |    1 |        74.42 ms |        67.61 ms |  12.13 tok/s | gap -9.2%，吞吐 +8.9%                |
| 1217 prompt + 8 out |    1 |        80.13 ms |        72.31 ms |   2.67 tok/s | gap -9.8%，吞吐 +2.3%，TTFT 基本不变 |

开启 replay cache 后收益集中在 decode：三档 decode `gap_p50` 均下降约 9%–10%，
prefill TTFT 差异小于 1%。长 prompt 场景的总吞吐受 prefill 时间支配，因此提升较小。
首次长 prompt 请求可能有一次性的冷路径开销，正式压测应在预热对应输入长度后计时。

### 7.3 `vllm bench serve` 压测

`vllm bench serve` 是独立的客户端命令，只连接已启动的 OpenAI 兼容服务，不需要再次加载模型。当前环境已安装与 DLC PyTorch 2.5.0 匹配的 `torchaudio` 和 `torchvision`，因此可以直接执行：

```bash
vllm bench serve --help=all
```

服务已启动在 `18005` 时，可用下面的随机数据集命令做客户端压测：

```bash
mkdir -p /root/workspace/work-docs/glm53flash/vllm-cl/docs/glm5_flash/bench_results
vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:18005 \
  --endpoint /v1/chat/completions \
  --model GLM-5.3-Flash-Mix \
  --tokenizer /data/models/GLM-5.3-Flash-Mix \
  --dataset-name random \
  --random-input-len 32000 \
  --random-output-len 512 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --ignore-eos \
  --temperature 0 \
  --seed 1234 \
  --save-result \
  --result-dir /root/workspace/work-docs/glm53flash/vllm-cl/docs/glm5_flash/bench_results
```

如果目标机误装了 CUDA 版 `torchaudio/torchvision`，应改回本次发布提供的 DLC wheel。仅在临时诊断、且无法安装 wheel 时，才可用 compat 目录兜底：

```bash
export PYTHONPATH=/root/workspace/work-docs/glm53flash/vllm-cl/docs/glm5_flash/compat${PYTHONPATH:+:$PYTHONPATH}
```

`vllm bench serve` 启动时出现的 `vllm._C`、`vllm._deepselect_C` 和 Triton 警告是 DLC 环境的预期提示；只要最终输出 `Serving Benchmark Result` 且 `Failed requests` 为 0，就说明客户端到模型的链路正常。客户端只需与已启动的服务通信，不需要重新加载模型。

## 8. 常见问题

| 现象                                                               | 处理                                                                                                                                                                                          |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 启动时报驱动 API 版本不一致                                        | 安装 `Driver Version_v26.0.1` 或与当前 Synapse 匹配的驱动；生产环境不要用 `DLC_SKIP_VERSION_CHECK=1` 绕过                                                                                 |
| `CRT TargetArch mismatch`                                        | 检查 `/usr/local/chipltech/simulator/lib/dlc_crt*.hex` 头 4 字节是否为 `0TYD`，确认 `DLC_TPU_VERSION=2` 与 `LLVM_PATH` 后在 `/root/workspace/CLsim` 执行 `./build.sh tyd --clean` |
| 导入 torchaudio/torchvision 失败                                   | 安装本次发布的 `torchaudio 2.5.0+dlc` 和 `torchvision 0.19.0+dlc` wheel；不要从 PyPI 安装 CUDA 版                                                                                         |
| `vllm bench serve` 报 `libcudart.so.13` 或缺少 `torchvision` | 当前环境被 CUDA 版媒体依赖覆盖；卸载后安装 DLC wheel。临时诊断可用 `PYTHONPATH=/root/workspace/work-docs/glm53flash/vllm-cl/docs/glm5_flash/compat` 兜底                                    |
| KV cache 容量不足                                                  | 降低 `--max-model-len` 或提高 `--gpu-memory-utilization`，并重新观察启动日志                                                                                                              |
| Graph 起不来但 eager 正常                                          | 删除 `--compilation-config`、`--kernel-config`，追加 `--enforce-eager`，确认模型和算子正常后再检查 graph capture size、KPool #19 和 metadata padding                                    |
| MTP 启动报 `KDA state: indices must be ...`                     | 确认 `vllm-cl` 已包含 KDA gather/scatter 连续性归一化修复；否则 `block_table[:,0]` 的 strided view 会被 kernel 拒绝                                                                       |
| MTP graph 请求中途 `synErrorLaunchFailure`                     | 确认 `vllm` 已包含 GDN spec padding 修复；该问题在 MTP batch 从 8 缩到 6、token 维 padding 到 16 时触发                                                                              |
| MTP 64 in-flight 但未同时 running                               | 当前 KDA state/KV 容量下单请求占用较大，实际 running 峰值约 20；这是容量限制，不是 graph bucket 丢失                                                                                    |
| 请求卡住                                                           | 查看服务日志，再用 `peek_stuck.sh` 定位算子；确认无残留进程后执行 `cltechpd_clnt -s --full`                                                                                               |
| 64 并发吞吐低于预期                                                | 检查请求是否落入较小 graph bucket；graph 不是所有并发都优于 eager                                                                                                                             |
| 首次请求很慢                                                       | Graph 捕获、KV 初始化和算子 warmup 在启动阶段完成；冷启动后再发请求，避免与启动并行压测                                                                                                       |

## 9. 生产建议

- 默认保持 `FULL_DECODE_ONLY`，不要为临时性能对比关闭 graph。
- MTP + graph 目前按验证/灰度路径使用；完成 AIME26、acceptance 和性能验收前，不要替换默认生产配置。
- 以 4096 上下文、0.80 显存利用率作为首轮验收基线，再按机器实际 KV 容量逐步放大。
- 对外服务使用 tmux、systemd 或容器编排托管，不要依赖交互式终端。
- 每次升级驱动、DLCSynapse、pytorch 或 kernel 库后，重新跑健康检查、单请求、8 并发和长上下文 smoke。
- 采集性能时固定请求长度、并发、`max_tokens`、温度和 graph 模式；否则 40-token bucket 等形状差异会导致不可比结果。
- 发布分支只保留生产代码与契约测试；探针、调试脚本和历史验证文档统一归档在 work-docs。
