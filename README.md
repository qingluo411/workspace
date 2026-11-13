# DLC / TYD 自研 AI 加速芯片软件栈

本仓库群提供 TYD 加速卡的完整推理软件栈：LLVM 编译器、CLsim、DLCSynapse
运行时、DLC_Custom_Kernel 算子库、DLCCL 集合通信、PyTorch DLC 后端，以及
vLLM / vllm-cl 推理服务插件。

当前已验证的重点模型是 **GLM-5.3-Flash-Mix**。适配、性能优化、NaN 排查和发布
整理已告一段落，发布代码位于 `vllm`、`vllm-cl`、`DLC_Custom_Kernel`、`pytorch`
四个仓库的 `release-glm53flash` / `release-glm53flash-2` / `release-glm53flash-3`
分支；设计、验证和交付文档统一归档在
`work-docs/glm53flash`。

> 本文最后更新：2026-11-12。详细启动步骤以
> `work-docs/glm53flash/vllm-cl/docs/glm5_flash/GLM-5.3-Flash_模型服务使用指南_2026-10-23.md`
> 为准。

## 当前发布矩阵

| 组件 | 分支 | 当前 HEAD | 说明 |
|---|---|---|---|
| `vllm` | `release-glm53flash` | `13b663fd09` | DLC vLLM fork；GLM/KDA graph padding 修正 |
| `vllm-cl` | `release-glm53flash` | `10a2fbb` | GLM 模型、算子、KV cache、Graph patch 的主要实现 |
| `DLC_Custom_Kernel` | `release-glm53flash-3` | `81f8e94a4` | GLM 融合算子、KDA 修复与 KPool #09 优化 |
| `pytorch` | `release-glm53flash-2` | `8182bdd8c3` | DLC 后端和 block 校验修复 |
| `DLCSynapse` | `main` | `e230fed5` | 运行时；Api_Version_26 |
| `CLsim` | `main` | `1e31b57` | 统一模拟器；TYD CRT ABI 3.2.0 |
| `DLC_CL` | `main` | `bc878f8` | DLCCL |
| `LLVM` | `main` | `cecb029aa019` | DLC 编译工具链；RT ABI 3.2.0 |

`DLCsim` 已退出当前构建和运行链，只保留源码仓库与旧安装目录
`/usr/local/chipltech/simulator.dlcsim-1b0504e2-20261111-092140` 作为回滚参考；
当前模拟器及 CRT 由 `CLsim` 提供。

`release-glm53flash` 系列（含 `-2` / `-3`）是当前交付线；早期 bring-up 特性分支只保留作历史记录，
后续修复和新版本应继续提交到 release 分支。GLM 的设计、验证、排障和交付文档统一
归档在 `work-docs/glm53flash`，不要重新放回 `vllm-cl` 发布分支。

运行环境：

- 8 × TYD，`/dev/cltech0..7`。
- 驱动 `Driver Version_v26.0.1` / `Api_Version_26`。
- 模拟器已切换到 CLsim `1e31b57`，系统安装的 CRT ABI 为 `3.2.0`。
- Python 3.10，`/root/workspace/.venv`。
- 当前工作区全部为 editable 源码安装：`torch 2.5.0`，`torch.dlc.is_available()`
  为 `True`；`torchaudio`/`torchvision` 模块版本分别为 `2.5.0+dlc` /
  `0.19.0+dlc`。
- 交付 wheel 的版本是 `vllm`/`vllm-cl 0.30.1rc0+dlc`、`torch 2.5.0`、
  `torchaudio 2.5.0+dlc`、`torchvision 0.19.0+dlc`。editable 安装的 dist
  metadata 会显示 `0.2.8.dev...`、`0.1.dev...`、`2.5.0a0...` 等源码开发版本，
  这是预期现象；不要用它替代 wheel 文件名或 `direct_url.json` 判断交付版本。

## GLM-5.3-Flash 适配结果

模型 checkpoint 为 `glm5_next` 混合架构：

- 45 层语言模型，hidden size 4096。
- 34 个 KDA linear-attention 层。
- 11 个 sparse MLA 层，层号为 `3, 7, 11, ..., 43`。
- index KPool：`index_kpool=4`、`index_topk=2048`、`index_n_heads=32`。
- 权重为 4-bit AWQ/group-size 128 混合精度 checkpoint，共 120 个分片，约
  178.81 GiB；TP=8 时每 rank 模型加载约 21.55 GiB。

已完成的生产代码路径包括：

- GLM5Next 模型、KDA、KPool indexer、sparse MLA 和权重映射 patch。
- `silu_and_mul_with_clamp`、`mhc_pre` / `mhc_post`、KDA RMSNorm sigmoid gate。
- KDA causal conv、chunk prefill、recurrent decode、state gather/scatter。
- KPool `#07` 更新/压缩、`#09` select+top-k+expand、`#19` slot mapping 融合链路。
- `sparse_attn_kv_cache` / Sparse MLA NoPE-512 路径。
- GLM 专用 KV cache 分组、Mamba/KDA state pool、metadata H2D 合并和 graph padding。
- `FULL_DECODE_ONLY` decode graph、持久 KPool graph buffer 和 Synapse replay cache。

生产路径默认：

```bash
export VLLM_PLUGINS=dlc,dlc_patches
export DLC_SKIP_VERSION_CHECK=0
export DLC_SYN_GRAPH_CACHE=1
export VLLM_USE_DLC_COL_MAJOR_MATMUL=1
export PYTORCH_DLC_ALLOC_CONF=expandable_segments:True

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

注意：`DLC_SYN_GRAPH_CACHE` 不是 graph capture 开关，而是 Synapse graph replay
的 kernel-batch 缓存。未设置时默认 `0`，graph 仍能运行，但每次 replay 都会重新
遍历和下发图节点。

冷启动约 3–4 分钟，主要耗时是模型权重加载、KV cache 初始化和 graph capture。
使用模型服务使用指南中的 `setsid` 命令时，服务日志写入
`/root/workspace/log/glm5_service.log`；健康检查为
`curl http://127.0.0.1:18005/health`。

## 已验证基线

默认验收配置：TP=8、`max_model_len=4096`、`max_num_batched_tokens=4096`、
`max_num_seqs=64`、`block_size=256`、`gpu_memory_utilization=0.80`、
`FULL_DECODE_ONLY`、`enable_jit_warmup=false`。

| 场景 | 结果 |
|---|---|
| 健康检查 | HTTP 200 |
| 单请求 20 prompt / 32 output | 2.107 s |
| 8 并发，每请求 20 / 32 | wall 3.751 s，约 68.24 output tok/s |
| 3610 prompt / 8 output | 72.419 s |
| 16K / 32K chunked prefill | 通过 |
| 1 / 2 / 8 / 16 / 24 / 32 并发 graph | 全部成功 |
| Graph 与 eager 输出 | 相同输入逐字一致 |
| Graph 捕获 | smoke 中 35 张 decode graph，最大 bucket 随 `max_num_seqs` 变化 |

性能参考：

- Graph vs eager：单请求 decode gap `110.27 → 67.64 ms`；8 并发吞吐约 `+9%`；
  32 并发基本持平；64 并发因 `38 → 40` bucket padding 约回退 5%。
- 开启 replay cache 后，单并发 decode gap 再下降约 9%–10%，prefill TTFT 基本不变。
- `DLC_SYN_GRAPH_CACHE=1` 单并发 decode 输出约 15–17.5 tok/s；1217-token prompt
  的端到端约 2.67 tok/s，主要时间消耗在 prefill。
- 历史性能分析中干净 decode 约 96.9 ms/token；单请求集合通信根因等待仍是最主要
  的残余瓶颈。

## 上下文长度上限

模型 checkpoint 声明 `max_position_embeddings=1,048,576`，但 8 卡 TYD 的实际
上限由 KV block 预算决定：

```text
blocks_per_request(L) = 69 + 2 × ceil(L / 256)
```

其中 69 为 34 个 KDA 层各 2 个 state block，加 1 个 KPool tail block。启动时
`blocks_per_request(max_model_len)` 不能超过日志报告的 KV block 总数。

| `gpu_memory_utilization` | `max-model-len` 理论上限 | 说明 |
|---:|---:|---|
| 0.80 | 171,776（约 171K） | 当前生产配置；满长时只有 1 个并发 |
| 0.95 | 约 232K，最高约 233K | 仅理论外推，只剩 5% 非 KV 余量，不建议生产使用 |

当前实际验证过的常规配置为 4096；131072 曾在 0.85 显存利用率、eager 路径下完成
长上下文验证。长上下文 prefill 时延会近似线性增长，容量上限不等于可接受时延。

## 已知限制

- 当前实际生产 runner 是 **V1**。V2 runner 需要 Triton，本机没有可用 Triton
  driver，启动时会自动回落到 V1。
- Graph 只覆盖 uniform decode；prefill 和 chunked prefill 仍为 eager。
- Graph 大 batch 已接近计算受限，收益被摊薄；非 2 的幂并发可能落入更大 bucket
  并承担 padding 成本。
- KPool `#09` 在长 prefill / 长上下文 decode 上慢于 native：单层交叉点约为
  prefill 180 token/请求、单请求 decode 448 token、32 请求 decode 800 token/请求。
  已按最终决策统一走 `#07 + #09`，不再启用按形状的 fast-path 门禁。
- TYD 不支持计算/通信重叠，集合通信同步等待仍是 decode 的主要残余开销之一。
- 长上下文性能主要受 prefill 支配；`max-model-len` 的容量上限不代表可接受的吞吐。
- AWQ MoE CPU fallback 依赖主机侧布局逆变换和反量化辅助函数，这些是生产必需代码。
- 0.95 显存利用率只有理论容量，未作为生产配置验证。

## 常见问题

| 现象 | 处理入口 |
|---|---|
| 输出全 `!` 或 NaN | 先读 NaN SOP；确认 KDA state pool 64 位地址修复已编入 kernel，且 warmup 后已清零 sparse MLA block 0 |
| 请求一直卡在 running | 先停服务，再用 `/usr/local/chipltech/cl_tools/CsrAccess/peek.sh -s status -t N` 和 `peek_stuck.sh -t N` 定位；确认无残留进程后执行 `cltechpd_clnt -s --full` |
| `vllm bench serve` 报 `libcudart.so.13` | 误装了 CUDA 版媒体依赖；安装发布的 DLC `torchaudio`/`torchvision` wheel |
| Graph 启动失败但 eager 正常 | 去掉 `--compilation-config`、`--kernel-config`，追加 `--enforce-eager` 先验证模型和算子 |
| 驱动 API 版本不一致 | 安装与 Synapse 匹配的 `Driver Version_v26.0.1`；生产不要用 `DLC_SKIP_VERSION_CHECK=1` 绕过 |
| `CRT TargetArch mismatch` | 检查 4 个 `dlc_crt*.hex` 头 4 字节是否为 `0TYD`，并确认 `DLC_TPU_VERSION=2`、`LLVM_PATH` 后在 `/root/workspace/CLsim` 执行 `./build.sh tyd --clean` |

## NaN 故障结论

GLM-5.3-Flash 曾出现全 `!` / NaN 输出，最终定位到两个相互独立的根因：

1. **KDA state pool 地址溢出**：DLC 目标下指针和 `unsigned long` 为 32 位，超过
   16 GiB 的 state 地址计算溢出，gather 实际未写入输出。已在
   `DLC_Custom_Kernel/dlc_kernels/attention/kda/kda_state_pool.h` 改为 64 位地址
   运算，并同时覆盖 gather/scatter。
2. **sparse MLA null block 被 warmup 污染**：V1 dummy sampler 用全 0 slot mapping
   执行 warmup，把 11 个 indexer 层的 block 0 写脏。已在
   `DLCWorker.compile_or_warm_up_model()` 的所有 warmup 后重新清零 block 0。

修复顺序验证很重要：只修 KDA 时首个 NaN 从 layer 0 移到 layer 3（首个 sparse MLA
层），说明前一根因已修复且暴露出第二根因。通用排查流程见 NaN SOP 文档。

## 构建与测试

源码构建使用：

```bash
bash /root/workspace/helper/compile.sh
```

该脚本依次构建 LLVM、CLsim、DLCSynapse、DLC_Custom_Kernel、DLC_CL、pytorch，
再 editable 安装 vLLM 和 vllm-cl。`helper/update.sh` 会清理并 `git pull origin main`，
**不要在有未提交发布改动的 release 分支上无检查执行**。

如需验证 CLsim 构建但不覆盖当前系统模拟器，可设置
`CLSIM_INSTALL_PREFIX=/tmp/clsim-stage`；只有 CLsim 会安装到该 staging 目录，
其余组件仍按脚本原有路径构建。

当前 wheel 目录：

- `wheelhouse/`：`torch-2.5.0`、`vllm-0.30.1rc0+dlc`、`vllm_cl-0.30.1rc0+dlc`
  及 `libcustom_dlc_perf_lib.so`。
- `wheelhouse-glm53flash/`：`torchaudio-2.5.0+dlc`、`torchvision-0.19.0+dlc`。
- `wheel.tar.gz`：上述 wheelhouse 的归档（不含 torchaudio/torchvision）。

`helper/get_package.sh` 是完整打包 recipe，但其中的 `DLC_INC`、`PY`、
`WHEELHOUSE` 等变量需要调用者提供；执行前应先检查脚本中的复制路径。

单独构建发布 wheel 的核心命令如下。先创建目录并切换 Python 环境：

```bash
export PATH=/root/workspace/.venv/bin:$PATH
export WHEELHOUSE=/root/workspace/wheelhouse
mkdir -p "$WHEELHOUSE"
```

PyTorch：

```bash
cd /root/workspace/pytorch
rm -rf build/* dist/* && USE_CUDA=0 python setup.py clean
PYTORCH_BUILD_VERSION=2.5.0 PYTORCH_BUILD_NUMBER=0 USE_CUDA=0 DEBUG=0 \
  python setup.py bdist_wheel --dist-dir "$WHEELHOUSE"
```

vLLM 与 vllm-cl：

```bash
cd /root/workspace/vllm
rm -rf build/* dist/* && python setup.py clean
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE='0.30.1rc0+dlc' \
VLLM_SKIP_VERSION_SUFFIX=1 MAX_JOBS=$(nproc) \
  python setup.py bdist_wheel --dist-dir "$WHEELHOUSE"

cd /root/workspace/vllm-cl
rm -rf build/* dist/* && python setup.py clean
SETUPTOOLS_SCM_PRETEND_VERSION='0.30.1rc0+dlc' MAX_JOBS=$(nproc) \
CMAKE_BUILD_TYPE=Release COMPILE_CUSTOM_KERNELS=1 \
  python setup.py bdist_wheel --dist-dir "$WHEELHOUSE"
```

torchaudio/torchvision 使用另一个产物目录。`DLC_INC` 指向已安装的 DLC 头文件；
完整参数以 `helper/get_package.sh` 为准：

```bash
export WHEELHOUSE=/root/workspace/wheelhouse-glm53flash
export DLC_INC=/usr/local/chipltech/synapse/include/external_includes
mkdir -p "$WHEELHOUSE"

cd /root/workspace/torchaudio-2.5
CXXFLAGS="-I$DLC_INC" CPLUS_INCLUDE_PATH="$DLC_INC" \
BUILD_VERSION='2.5.0+dlc' PYTORCH_VERSION='2.5.0' USE_CUDA=0 \
BUILD_CUDA_CTC_DECODER=0 USE_FFMPEG=0 BUILD_SOX=0 MAX_JOBS=8 \
CMAKE_BUILD_PARALLEL_LEVEL=8 \
  python setup.py bdist_wheel --dist-dir "$WHEELHOUSE"

BUILDDEPS=$(mktemp -d /tmp/torchvision-builddeps.XXXXXX)
python -m pip install --target "$BUILDDEPS" 'setuptools<81' wheel
cd /root/workspace/torchvision
PYTHONPATH="$BUILDDEPS${PYTHONPATH:+:$PYTHONPATH}" \
CXXFLAGS="-I$DLC_INC" CPLUS_INCLUDE_PATH="$DLC_INC" \
BUILD_VERSION='0.19.0+dlc' PYTORCH_VERSION='2.5.0' \
FORCE_CUDA=0 FORCE_MPS=0 TORCHVISION_USE_PNG=0 TORCHVISION_USE_JPEG=0 \
TORCHVISION_USE_NVJPEG=0 TORCHVISION_USE_FFMPEG=0 \
TORCHVISION_USE_VIDEO_CODEC=0 MAX_JOBS=8 \
  python setup.py bdist_wheel --dist-dir "$WHEELHOUSE"
```

打包到其他机器时还必须带上 `libcustom_dlc_perf_lib.so`，并安装到目标机的
`/usr/local/chipltech/synapse/lib`；wheel 本身不包含该 kernel 库。

测试基线：

- `python -m pytest /root/workspace/vllm-cl/tests -q`：
  GLM 回归为 `161 passed, 10 subtests passed`。
- `python -m compileall`、`git diff --check`、变更文件 `ruff check/format` 通过。
- vLLM graph padding 用例：
  `tests/v1/attention/test_gdn_metadata_builder.py` 的新增用例已独立等价验证通过。
- 服务 smoke：健康检查、单请求、8 并发、3610-token 长上下文均通过。

## 关键文档

模型相关文档统一位于
`work-docs/glm53flash/vllm-cl/docs/glm5_flash/`：

- `GLM-5.3-Flash_模型服务使用指南_2026-10-23.md`：对外交付、启动参数、性能与
  上下文上限。
- `GLM-5.3-Flash_代码清理与合并说明_2026-10-23.md`：发布代码改动、测试结果和
  合入主分支注意事项。
- `GLM-5.3-Flash_Graph捕获验证_2026-10-22.md`：Graph 修复、并发矩阵、graph
  replay cache 复测与已知 bucket 限制。
- `GLM-5.3-Flash_NaN问题排查总结与SOP_2026-10-22.md`：KDA 地址溢出、null block
  污染、证据链和通用 NaN SOP。
- `GLM-5.3-Flash_发布前功能与性能验证_2026-10-22.md`：功能矩阵、快路径命中和
  M1–M8 性能指标。
- `KPool_09_select_topk_性能量化_2026-10-22.md`：`#09` 与 native 的交叉点、
  长上下文代价和最终门禁决策。
- `GLM-5.3-Flash_vLLM-CL_联调问题与修复记录.md`、`GLM-5.3-Flash_vLLM-CL_适配研究.md`
  和 `GLM-5.3-Flash_vLLM-CL_阻断点2-7.md`：早期适配与阻断点记录。
- `start_glm5_flash_vllm_cl.sh`：内部启动/排障脚本；正式交付以 `vllm serve`
  命令行为准，脚本不是部署依赖。

## 仓库目录

```text
LLVM/              DLC 编译器工具链
CLsim/             统一模拟器 + TYD CRT 固件
DLCsim/            旧模拟器仓库，仅保留回滚，不参与当前构建
DLCSynapse/        Synapse 运行时、CUDA-like API、Graph、profiler
DLC_Custom_Kernel/ AOT kernel 库与分发元信息
DLC_CL/            集合通信 DLCCL
pytorch/           PyTorch DLC 后端
vllm/              vLLM DLC fork
vllm-cl/           vLLM DLC plugin 与 GLM 适配
torchaudio-2.5/    DLC torchaudio 源码
torchvision/       DLC torchvision 源码
helper/            compile/update/status 辅助脚本
work-docs/         设计与验证文档归档（不进入 vllm-cl 发布分支）
work-tests/        临时端到端测试脚本
wheelhouse*/       发布 wheel 产物
```
