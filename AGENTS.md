# AGENTS.md — 自研 AI 加速芯片（DLC / TYD）软件栈速查

> 本文档用于快速回忆本仓库群（`/root/workspace`）的全貌：芯片、分层架构、
> 各仓库职责、构建顺序、运行时调用链与当前环境状态。
>
> 最后更新：2026-10-27。当前重点是 **GLM-5.3-Flash-Mix** 的
> `release-glm53flash` 发布线；仓库 HEAD、驱动、发布结论和文档入口见下表与第 7/8 节。
> 面向人阅读的项目总览见根目录 `README.md`。

---

## 0. 一句话总览

这是一个 **类 Habana Gaudi / CUDA 风格** 的自研 AI 加速芯片软件栈：
自研 LLVM 后端负责把 kernel 编成芯片二进制 → DLCsim 提供指令级模拟器与 CRT 固件 →
DLCSynapse 是运行时核心（CUDA-like `dlc_*` API + Graph `synapse_*` API）→
PyTorch 以 `device='dlc'` 接入 → vLLM 通过硬件插件跑推理。

---

## 1. 硬件层

| 项目 | 值 |
|---|---|
| 卡型号 | **TYD**（真机） |
| `hlthunk_device_name` 枚举 | `HLTHUNK_DEVICE_DLC=8`, `HLTHUNK_DEVICE_TYD=9`, `HLTHUNK_DEVICE_HHP=10`（定义在 `DLCSynapse/dlc_thunk/include/hlthunk.h`） |
| 卡数量 | 8 张，`/dev/cltech0..7`（主设备号 508） |
| sysfs | `/sys/devices/virtual/chipltech/cltech0..7` |
| 内核驱动 | `chipltech.ko`（已加载；路径 `/work/chipltech.ko`，由 `DLCKERNELDRIVER_PATH=/work/` 指向；源码不在 workspace） |
| 驱动版本 | `Driver Version_v26.0.1`，`Api_Version_26`（GLM-5.3-Flash 发布验证组合） |
| 固件 | `bluejay/build/dlc_init/bluejay_nrz.fw.1.2.28.bin` + HBM 训练序列等 |
| PCI | `0000:d5:00.0` |

### 片上计算 / 存储层级

- 每颗芯片 **2 个 XYS core**：`xys0` / `xys1`。
  XYS = "Xuan Yuan Sword"，内含标量处理器 + 向量处理器 + 矩阵单元（`simulator/include/xys/Xys.h`）。
- 存储层级（`Veloce_Architecture.h` 与 `synapse_api.h` 的 `SynDeviceMemFlag` 一致）：

| 层级 | 容量 | xys0/xys1 各一份 |
|---|---|---|
| IMEM | ≈680 KB (`0xAAA00`) | ✅ |
| SMEM | 1 MB (`0x100000`) | ✅ |
| VMEM | 16 MB (`0x1000000`) | ✅ |
| CMEM | 32 MB (`0x2000000`) | —（共享） |
| HBM | 64 GB (`0x1000000000`) | —（共享） |

- 命令行/调试工具位于 `/usr/local/chipltech/cl_tools/`（CSR 访问、HBM 训练序列、ICC init 等）。
- 固件初始化脚本：`/usr/local/chipltech/bluejay/dlc_init.sh`。

---

## 2. 软件栈分层（自底向上）

```
┌─────────────────────────────────────────────────────────────┐
│  应用/服务    vllm (+ vllm-cl 插件)  /  训练脚本              │
├─────────────────────────────────────────────────────────────┤
│  框架         pytorch (fork, device='dlc', torch.dlc)         │
│               aten/src/ATen/dlc/ + c10 DeviceType::DLC=21     │
├─────────────────────────────────────────────────────────────┤
│  算子库       DLC_Custom_Kernel → libcustom_dlc_perf_lib.so   │
│  通信库       DLC_CL           → libdlccl.so (类 NCCL)        │
├─────────────────────────────────────────────────────────────┤
│  运行时       DLCSynapse       → libsynapse_core/backend.so   │
│               dlc_runtime      → libdlc_runtime.so            │
│               （提供 CUDA-like dlc_* API + Graph synapse_*）  │
├─────────────────────────────────────────────────────────────┤
│  模拟器       DLCsim           → libsimulator.so + CRT hex    │
│  编译器       LLVM (dlc target) → clang/ld.lld/llvm-mc/elf2hex │
├─────────────────────────────────────────────────────────────┤
│  用户态驱动   libhl-thunk.so (DLCSynapse/dlc_thunk)           │
├─────────────────────────────────────────────────────────────┤
│  内核驱动     chipltech.ko  +  /dev/cltech*                   │
└─────────────────────────────────────────────────────────────┘
```

**关键机制：真机 vs 模拟的切换**

- 真机：`libhl-thunk.so → /dev/cltech*`
- 模拟：`libsimulator.so`（指令级解释器）
- 切换开关：环境变量 **`DLC_SYN_USE_SIM=1`**，运行期读取（`DLCSynapse/external_includes/dlc/utils.hpp:386 syn_use_sim()`）。
- 两条路都编进同一个 `libsynapse_*.so`，因此 PyTorch 的 `.so` 同时依赖 `libsimulator.so` 和 `libhl-thunk.so`。
- `DLC_SYN_USE_SIM=1` 会隐含 `DLC_SYN_BLOCKING=1`。

---

## 3. 各仓库职责、产物与 HEAD

| 仓库 | 分支 | HEAD | 职责 / 当前状态 |
|---|---|---|---|
| **LLVM** | `main` | `2fb2abd9e629` | DLC TPU 编译器后端（`--target=dlc`、`-mcpu=dlc-gen2`、`-std=dc99`） |
| **DLCsim** | `main` | `c6e2808d` | 指令级模拟器 + CRT；`lib-rt/*.hex` 为本地 TYD 生成改动，未提交 |
| **DLCSynapse** | `main` | `37d6c60d` | Synapse 运行时、Graph、内存、profiler、用户态驱动 |
| **DLC_Custom_Kernel** | `release-glm53flash` | `795e8ca3a` | AOT kernel 库；包含 GLM 融合算子与 KDA 地址溢出修复 |
| **DLC_CL** | `main` | `b4e3423` | DLCCL 集合通信 |
| **pytorch** | `release-glm53flash` | `2ff2142cb9` | `device='dlc'` 后端；包含 block 校验修复 |
| **vllm** | `release-glm53flash` | `8cfd356105` | vLLM DLC fork；KDA graph padding 修正 |
| **vllm-cl** | `release-glm53flash` | `253bd69` | GLM 模型、算子、KV cache、Graph patch 主实现；工作区仅 `vllm_cl/_version.py` 为生成式改动 |

> 所有仓库 remote 均为 `git@github.com:ChipLTech/<repo>.git`。
> `torchaudio-2.5/`、`torchvision/` 为 DLC 媒体依赖源码，发布 wheel 已在
> `wheelhouse-glm53flash/`；不要用 PyPI 的 CUDA 版覆盖。
>
> `release-glm53flash` 是当前唯一交付线；早期 bring-up 特性分支只作历史参考。
> 新修复继续提交到 release 分支，模型文档继续归档在 `work-docs/glm53flash`。

### 官方依赖图（`DLC_Custom_Kernel/README.md`）

```
LLVM ──┬─→ DLCsim ──┬─→ DLCSynapse ──┬─→ DLC_Custom_Kernel ──┐
       │            │               ├─→ DLC_CL              ├─→ pytorch ─→ vllm ─→ vllm-cl
       └────────────┴───────────────┘                       │
DLC-Kernel-Driver ─────────────────→ DLCSynapse              ┘
```

`helper/compile.sh` 按此顺序构建并 editable 安装；`helper/update.sh` 会先
`git restore/clean` 再 `git pull origin main`，不要在带发布改动的 release 分支上
无检查执行。

---

## 4. 构建顺序与产物布局

### 4.1 构建脚本

```bash
bash /root/workspace/helper/compile.sh   # 全栈源码构建 + pytorch develop + editable vllm/vllm-cl
bash /root/workspace/helper/check_status.sh
```

- `helper/compile.sh`：日常源码重建入口。
- `helper/update.sh`：拉取各仓库 `main` 的辅助脚本；会清理工作区，不适合 release
  分支直接执行。
- `helper/get_package.sh`：打包 recipe，需调用者提供 `DLC_INC`、`PY`、
  `WHEELHOUSE` 等变量；执行前检查复制路径。
- `wheelhouse/`：torch/vLLM/vllm-cl 和 `libcustom_dlc_perf_lib.so`。
- `wheelhouse-glm53flash/`：`torchaudio-2.5.0+dlc`、`torchvision-0.19.0+dlc`。

单独重建 wheel 时，统一把 `/root/workspace/.venv/bin` 放到 `PATH` 前部，并显式指定
`--dist-dir`：

```bash
export PATH=/root/workspace/.venv/bin:$PATH
export WH=/root/workspace/wheelhouse
mkdir -p "$WH"

cd /root/workspace/pytorch
rm -rf build/* dist/* && USE_CUDA=0 python setup.py clean
PYTORCH_BUILD_VERSION=2.5.0 PYTORCH_BUILD_NUMBER=0 USE_CUDA=0 DEBUG=0 \
  python setup.py bdist_wheel --dist-dir "$WH"

cd /root/workspace/vllm
rm -rf build/* dist/* && python setup.py clean
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE='0.30.1rc0+dlc' \
VLLM_SKIP_VERSION_SUFFIX=1 MAX_JOBS=$(nproc) \
  python setup.py bdist_wheel --dist-dir "$WH"

cd /root/workspace/vllm-cl
rm -rf build/* dist/* && python setup.py clean
SETUPTOOLS_SCM_PRETEND_VERSION='0.30.1rc0+dlc' MAX_JOBS=$(nproc) \
CMAKE_BUILD_TYPE=Release COMPILE_CUSTOM_KERNELS=1 \
  python setup.py bdist_wheel --dist-dir "$WH"
```

`torchaudio`/`torchvision` 另用 `WH=/root/workspace/wheelhouse-glm53flash`、
`DLC_INC=/usr/local/chipltech/synapse/include/external_includes`，完整环境变量见
`helper/get_package.sh`。搬机时别忘了单独复制
`libcustom_dlc_perf_lib.so` 到 `/usr/local/chipltech/synapse/lib`。

### 4.2 安装前缀

| 组件 | 安装前缀 |
|---|---|
| DLCsim | `/usr/local/chipltech/simulator` |
| DLCSynapse | `/usr/local/chipltech/synapse` |
| DLC_CL | `/usr/local/chipltech/dlccl` |
| DLC_Custom_Kernel | 安装到 `/usr/local/chipltech/synapse/lib` |
| bluejay / cl_tools | `/usr/local/chipltech/bluejay`、`/usr/local/chipltech/cl_tools` |

### 4.3 kernel 的编译链路（AOT，关键机制）

见 `DLC_Custom_Kernel/build/build.ninja` 的真实命令：

```
clang -S --target=dlc -std=dc99 -mcpu=dlc-gen2 -DDLC_TPU_VERSION=2 ...
  ↓
llvm-mc --arch=dlc -filetype=obj          → <kernel>.o       （xys0）
llvm-mc --arch=dlc -filetype=obj -mattr="+core1"
                                          → <kernel>_xys1.o  （xys1）
  ↓
ld.lld --entry=main                       → 链接后的 dlc ELF
  ↓
llvm-objcopy -I binary -O elf64-x86-64    → 当作数据段嵌进 x86 .o
  ↓
链接进 libcustom_dlc_perf_lib.so
```

- 每个 kernel 都产出 **xys0 / xys1 两份**，并生成 `*.dlcasm`（`llvm-objdump` 反汇编）留档。
- 元信息：`dlc_src/kernel_info.yaml` 声明 kernel；`dlc_src/gen.py` 生成
  `dlc_src/enabled_kernels.hpp`（`DispatchType::DLC/CPU/BOTH`，决定 PyTorch 走 DLC 还是回退 CPU）。
- 运行期：`DLCSynapse/synapse_backend/utils/chip_arch.cpp` 用
  `dlopen("libcustom_dlc_perf_lib.so")` 取 `TargetArch / KernelCount / KernelVersion / RTABIVersion / KernelList` 等符号。

---

## 5. 运行时调用链

```
torch.ones(5, device='dlc')
  └─ torch.dlc / aten/src/ATen/dlc/EmptyTensor.cpp
       └─ libc10_dlc.so / libtorch_dlc.so → dlc_runtime_api.h
            └─ libsynapse_core.so (synapse_core/synapse_core.cpp)
                 ├─ 初始化：读 CRT hex → CheckCrtTargetArch() 与真实芯片比对
                 ├─ 真机：libhl-thunk.so → /dev/cltech0
                 └─ 模拟：libsimulator.so（XYS 解释器）
```

### 5.1 两套 API（`DLCSynapse/external_includes/`）

- `dlc_runtime_api.h`：CUDA 风格（`dlcMalloc` / `dlcMemcpy` / `dlcLaunchKernel` / stream / event）。
  文件头注释写着 "Wrapper over synapse api making it same as dlc runtime api"。
- `synapse_api.h`：Graph 风格（`synStream*` / `synDevice*`，类 Habana SynapseAI）。
- PyTorch 主要用前者；`synapse_api.h` 用于图/低层控件。

### 5.2 CRT 固件与 TargetArch 标记（排错必读）

- CRT = 每颗核启动时执行的引导代码，源码在 `DLCsim/lib-rt/`（`crt0.S`、`crt0_xys1.S`、`crt_sync.S`…），
  编出 **4 个 hex**：`dlc_crt.hex`、`dlc_crt_xys1.hex`、`dlc_crt_cmem.hex`、`dlc_crt_cmem_xys1.hex`。
- hex 头 4 字节是 **TargetArch 标记**：
  - `0DLC` → DLC（TargetArch=1）
  - `0TYD` → TYD（TargetArch=2）
  - `0HHP` → HHP（TargetArch=3）
- 标记由 `DLCsim/lib-rt/crt0.S` 中的 `DLC_TPU_VERSION` 宏控制：
  `DLC_TPU_VERSION=1` 生成 `0DLC`，`=2` 生成 `0TYD`，`=3` 生成 `0HHP`。
- 初始化时 `DLCSynapse/synapse_core/synapse_core.cpp:243 CheckCrtTargetArch()`
  会把 CRT 标记与真实芯片（`hlthunk` 返回的设备名）比对，不一致直接报错：
  ```
  CRT TargetArch mismatch: current real device chip is TYD (TargetArch=2),
  but .../dlc_crt.hex was built for DLC (TargetArch=1)
  ```
- **常见坑**：DLCsim 的 CRT 只有在该仓库 CMake 配置时能看到 `LLVM_PATH`
  （`DLCsim/CMakeLists.txt:40-45`）才会重新生成；否则静默跳过，只重装旧的 hex。
  → 重新编译 CRT 必须确保：
  ```bash
  export LLVM_PATH=/root/workspace/LLVM
  export DLC_TPU_VERSION=2       # 目标 TYD
  cd /root/workspace/DLCsim && rm -rf build && ./build.sh
  ```
  验证：
  ```bash
  for f in dlc_crt.hex dlc_crt_xys1.hex dlc_crt_cmem.hex dlc_crt_cmem_xys1.hex; do
    printf '%s: ' "$f"; head -c 4 /usr/local/chipltech/simulator/lib/$f; echo
  done   # 应输出 0TYD
  ```
- **RTABI 版本**：`dlc_crt.rtabi = 3.0.0`，clang 内建也报 "DLC RT ABI version: 3.0.0"，
  用于编译期/运行期 ABI 对齐（`DLCSynapse/synapse_backend/utils/rtabi_version.*`）。

### 5.3 DLCSynapse 关键源码位置（`DLCSynapse/`）

| 路径 | 内容 |
|---|---|
| `synapse_core/synapse_core.cpp` | API 实现、CRT 校验、kernel 入队 |
| `synapse_core/synapse_graph*.cpp` | 图捕获 / 执行 |
| `synapse_backend/low_level_driver/runtime.cpp` | 设备发现、真机/模拟分派 |
| `synapse_backend/low_level_driver/driver_device.cpp` | 真机（hlthunk）路径 |
| `synapse_backend/low_level_driver/simulator_loader.*` | 模拟器 dlopen 与函数指针表 |
| `synapse_backend/low_level_driver/simulator_ops.hpp` | 模拟器操作封装 |
| `synapse_backend/utils/chip_arch.cpp` | TargetArch / custom kernel 库加载 |
| `dlc_runtime/dlc_runtime.cpp` | CUDA-like runtime API 实现 |
| `dlc_thunk/libhlthunk.c` | 用户态驱动（→ `libhl-thunk.so`） |
| `synapse_perf/` | profiler（`syn_perf`，Chrome Tracing JSON） |
| `cltech_python/` | Cython 绑定（`cltech`），当前未安装 |

---

## 6. 环境变量速查

| 变量 | 值 / 含义 |
|---|---|
| `DLC_TPU_VERSION` | `2` = TYD（本机）；`1`=DLC，`3`=HHP。编译期/运行期都要一致 |
| `LLVM_PATH` | `/root/workspace/LLVM`（**仅写在 `.bashrc`，非交互 shell 看不到！**） |
| `LD_LIBRARY_PATH` | `dlccl/lib : simulator/lib : synapse/lib : kernel : bluejay/build` |
| `CUSTOM_KERNEL_PATH` | `/usr/local/chipltech/synapse/lib` |
| `DLCKERNELDRIVER_PATH` | `/work/`（内核驱动 `chipltech.ko` 所在） |
| `DLC_SYN_USE_SIM` | `1` → 用 DLCsim 而非真机（隐含 `DLC_SYN_BLOCKING=1`） |
| `DLC_SYN_BLOCKING` | 阻塞所有 `dlc_runtime_api` 调用 |
| `DLC_SYN_DEBUG` / `DLC_SYN_VERBOSE` | 日志；`VERBOSE=3` 打印算子参数 |
| `DLC_SKIP_VERSION_CHECK` | 生产必须为 `0`；仅隔离旧镜像临时设为 `1` |
| `DLC_SYN_GRAPH_CACHE` | `1` = 缓存 graph replay 的 kernel queue batch；默认 `0`，不是 graph capture 开关 |
| `VLLM_PLUGINS` | GLM 生产必须为 `dlc,dlc_patches` |
| `VLLM_USE_DLC_COL_MAJOR_MATMUL` | `1` = 使用适配 checkpoint 的 column-major linear 路径 |
| `PYTORCH_DLC_ALLOC_CONF` | `expandable_segments:True`；缓解 Mamba/MLA 交错分配碎片 |
| `VLLM_LOGGING_LEVEL` | 生产默认 `INFO`，避免 DEBUG 日志淹没 |
| `DLC_SIM_ENHANCED_*` / `DLC_SIM_ENABLE_*` | 模拟器各类调试打印（见 `DLCsim/setenv.sh` 全表） |

> 注意：`.bashrc` 与 `/etc/environment` 不一致会导致交互/非交互行为不同。

---

## 7. 当前环境状态（2026-10-27）

| 组件 | 状态 |
|---|---|
| LLVM / DLCsim / DLCSynapse / DLC_CL | 已构建并安装；DLCsim CRT 为 TYD |
| DLC_Custom_Kernel | 已构建并安装，`libcustom_dlc_perf_lib.so` 约 1.4 GB |
| pytorch | 已 editable 安装；`import torch` 正常，`torch 2.5.0`，DLC 后端可用 |
| vLLM / vllm-cl | editable 源码安装；交付 wheel 版本 `0.30.1rc0+dlc`，V1 runner |
| torchaudio / torchvision | 当前 editable 源码模块版本 `2.5.0+dlc` / `0.19.0+dlc`；交付 wheel 同版本 |
| 驱动 | `Driver Version_v26.0.1`、`Api_Version_26` |
| 服务 | 当前未运行，`18005` 空闲 |

- Python 环境：`/root/workspace/.venv`，Python 3.10。
- 真机 8 卡 `/dev/cltech0..7` 可见；`device_init_flag` 正常。
- `torch.dlc.is_available()` 为 `True`。
- 按使用指南的 `setsid` 方式启动时，服务日志默认在
  `/root/workspace/log/glm5_service.log`；前台启动则直接输出到终端。
- editable 安装的 dist metadata 会显示源码开发版本（如 vLLM `0.2.8.dev...empty`、
  vllm-cl `0.1.dev...`、torchaudio `2.5.0a0...`），不是交付 wheel 版本；判断部署
  版本应看 wheel 文件名、`direct_url.json` 或模块版本文件。
- 本机没有可用 Triton driver；vLLM 会提示 V2 runner 不可用并回落 V1，这是当前预期。
- 本地生成改动：`DLCsim/lib-rt/*.hex`、`vllm-cl/vllm_cl/_version.py`；提交前应确认是否需要纳入。

---

## 8. GLM-5.3-Flash 发布状态（2026-10-27）

### 8.1 模型与生产路径

- 模型：`/data/models/GLM-5.3-Flash-Mix`，`glm5_next`，45 层、hidden 4096。
- 34 个 KDA linear-attention 层；11 个 sparse MLA 层（3, 7, 11, …, 43）。
- `index_kpool=4`、`index_topk=2048`、`index_n_heads=32`。
- checkpoint 约 178.81 GiB、120 分片；TP=8 每 rank 加载约 21.55 GiB。
- 实际生产 runner 为 V1；prefill/chunked prefill 为 eager，decode 使用
  `FULL_DECODE_ONLY` graph。

已进入生产路径的 GLM 代码：

- GLM5Next 模型/indexer/sparse MLA/权重映射 patch。
- `silu_and_mul_with_clamp`、`mhc_pre/post`、KDA RMSNorm sigmoid gate。
- KDA causal conv、chunk prefill、recurrent decode、state gather/scatter。
- KPool `#07` append+compress、`#09` select/top-k/expand、`#19` slot mapping。
- DLC 专用 KV cache 分组、Mamba/KDA state pool、metadata H2D 合并、graph padding。
- `DLC_SYN_GRAPH_CACHE=1` 的 cached replay、KPool 持久 graph buffer。

### 8.2 启动基线

```bash
export VLLM_PLUGINS=dlc,dlc_patches
export DLC_SKIP_VERSION_CHECK=0
export DLC_SYN_GRAPH_CACHE=1
export VLLM_USE_DLC_COL_MAJOR_MATMUL=1
export PYTORCH_DLC_ALLOC_CONF=expandable_segments:True

vllm serve /data/models/GLM-5.3-Flash-Mix \
  --served-model-name GLM-5.3-Flash-Mix --language-model-only --dtype bfloat16 \
  --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 64 \
  --block-size 256 --tensor-parallel-size 8 --gpu-memory-utilization 0.80 \
  --trust-remote-code --port 18005 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --kernel-config '{"enable_jit_warmup": false}'
```

冷启动约 3–4 分钟。启动日志应出现 `Application startup complete`、graph capture、
KV cache 容量和 `FULL_DECODE_ONLY`。

### 8.3 验证与性能摘要

- 健康检查、单请求、8 并发、16K/32K chunked prefill、prefill→decode 边界通过。
- 1/2/8/16/24/32 并发 graph 全部成功；graph 与 eager 同输入逐字一致。
- smoke：单请求 20/32 为 2.107 s；8 并发 20/32 为 3.751 s、约 68.24 tok/s；
  3610-token prompt 为 72.419 s。
- Graph vs eager：单请求 decode gap `110.27 → 67.64 ms`；8 并发吞吐 +9%；
  32 并发持平；64 并发因 `38 → 40` bucket padding 回退约 5%。
- Graph replay cache=1 vs 0：单并发 decode gap 再降约 9%–10%，TTFT <1% 变化。
- 生产配置单并发 decode 约 15–17.5 tok/s；长 prompt 端到端受 prefill 支配。
- 历史干净 decode 约 96.9 ms/token；残余大项是集合通信 root-critical 等待。
- vllm-cl 定向测试：`161 passed, 10 subtests passed`；compileall/ruff/diff-check 通过。

### 8.4 上下文长度

- 模型声明上限：1,048,576。
- 单请求 KV block：`69 + 2 × ceil(L / 256)`；69 = 34 个 KDA 层 × 2 个 state block
  + 1 个 KPool tail block。
- 0.80 显存利用率：1,411 blocks，理论上限 **171,776（约 171K）**，满长并发 1.00x。
- 0.95 显存利用率：约 1,883–1,891 blocks，理论 **约 232K，最高约 233K**；风险高，
  未作为生产配置。
- 常规验证值为 4096；131072 曾在 0.85/eager 下完成长上下文验证。长上下文 prefill
  会近似线性变慢。

### 8.5 已知限制

- 无 Triton，V2 runner 不可用；V1 是当前生产路径。
- Graph 只覆盖 decode；64 并发及非 2 次幂并发存在 bucket padding 成本。
- KPool `#09` 在长 prefill/长上下文 decode 上慢于 native；交叉点约为 prefill
  180 token、单请求 decode 448 token、32 请求 decode 800 token/请求。最终决策是
  不按形状设门禁，prefill/decode 统一走 `#07 + #09`。
- 硬件不支持计算/通信重叠；集合通信等待仍是主要优化空间。
- AWQ MoE CPU fallback 的主机布局逆变换/反量化辅助函数是生产必需，不是调试代码。
- 发布文档只归档在 `work-docs/glm53flash`，不要重新拷回 vllm-cl 发布分支。

### 8.6 NaN 故障的两个根因

1. **KDA state pool 地址溢出**：DLC 下指针/`unsigned long` 为 32 位，超过 16 GiB
   的地址计算溢出，gather 未写入输出。修复在
   `DLC_Custom_Kernel/dlc_kernels/attention/kda/kda_state_pool.h`，gather/scatter
   同时改为 64 位地址运算。
2. **sparse MLA null block 污染**：V1 dummy warmup 以全 0 slot mapping 执行，
   污染 11 个 indexer 层的 block 0。修复在
   `vllm-cl/vllm_cl/worker/dlc_worker.py::compile_or_warm_up_model()`，所有 warmup
   后重新清零 block 0。

只修 KDA 时首个 NaN 从 layer 0 移到 layer 3，说明两个根因串联。完整证据链与 SOP
见第 10 节文档索引。

### 8.7 文档入口

统一目录：`work-docs/glm53flash/vllm-cl/docs/glm5_flash/`

- 对外部署：`GLM-5.3-Flash_模型服务使用指南_2026-10-23.md`
- 代码清理/合入：`GLM-5.3-Flash_代码清理与合并说明_2026-10-23.md`
- Graph：`GLM-5.3-Flash_Graph捕获验证_2026-10-22.md`
- NaN/SOP：`GLM-5.3-Flash_NaN问题排查总结与SOP_2026-10-22.md`
- 发布前验证：`GLM-5.3-Flash_发布前功能与性能验证_2026-10-22.md`
- KPool #09：`KPool_09_select_topk_性能量化_2026-10-22.md`
- 早期适配：`GLM-5.3-Flash_vLLM-CL_联调问题与修复记录.md`、
  `GLM-5.3-Flash_vLLM-CL_适配研究.md`、`GLM-5.3-Flash_vLLM-CL_阻断点2-7.md`

---

## 9. 常见问题排查入口

| 现象 | 先查 |
|---|---|
| `CRT TargetArch mismatch` | 第 5.2 节：`head -c 4` 检查 4 个 hex；确认 `DLC_TPU_VERSION` 与 `LLVM_PATH` |
| `import torch` 找不到 `.so` | `pytorch/torch/lib/` 是否为空 → 重跑 `setup.py develop` |
| 找不到 `libsimulator.so` / `libhl-thunk.so` | `LD_LIBRARY_PATH` 是否包含 `simulator/lib`、`synapse/lib` |
| 真机连不上 | `ls /dev/cltech*`、`lsmod \| grep chipltech`、`cat /sys/devices/virtual/chipltech/cltech0/version` |
| kernel 未生效/回退 CPU | `dlc_src/enabled_kernels.hpp` 的 `DispatchType`、`libcustom_dlc_perf_lib.so` 是否最新 |
| 想只用模拟器 | `export DLC_SYN_USE_SIM=1` |
| 性能/时序分析 | `synapse_perf/syn_perf` + `DLCsim/profiler` |
| GLM 输出全 `!` / NaN | 先读 NaN SOP；检查 KDA 64 位地址修复是否已编进 kernel，warmup 后是否重新清零 sparse MLA block 0 |
| Graph 启动失败 / `query_start_loc must cover` | 检查 vLLM KDA/GDN padding 修复、KPool #19 持久 slots、graph bucket padding |
| `vllm bench serve` 报 `libcudart.so.13` | 安装 DLC `torchaudio/torchvision` wheel，不要用 CUDA 版；临时可加 compat PYTHONPATH |
| 请求卡在 running | 停服务，`peek.sh -s status -t N`、`peek_stuck.sh -t N`；确认无占用进程后 `cltechpd_clnt -s --full` |
| 长上下文吞吐退化 | 先看 prefill 时间；#09 在长 prefill/长上下文 decode 已知慢于 native，当前不做形状门禁 |

---

## 10. 目录速查（`/root/workspace`）

```
DLCSynapse/        运行时核心（synapse_core / synapse_backend / dlc_runtime / dlc_thunk / synapse_perf）
DLCsim/            模拟器 + CRT + server（lib-rt/、device/、src/xys、profiler/）
DLC_Custom_Kernel/ 2686 个 AOT kernel（dlc_kernels/）+ 分发元信息（dlc_src/）
DLC_CL/            集合通信 DLCCL
LLVM/              DLC 编译器工具链（dlc target）
pytorch/           PyTorch fork（device='dlc'）
vllm/              vLLM fork
vllm-cl/           vLLM DLC hardware plugin
torchaudio-2.5/    DLC torchaudio 源码
torchvision/       DLC torchvision 源码
helper/            compile/update/check_status/get_package 辅助脚本
work-docs/         设计与验证文档归档（glm53flash/...）
work-tests/        临时端到端测试脚本
wheelhouse/        torch/vLLM/vllm-cl/kernel 产物
wheelhouse-glm53flash/ torchaudio/torchvision DLC wheel
pyproject.toml     uv 管理的 workspace 元数据
README.md          项目总览与 GLM 发布状态
```
