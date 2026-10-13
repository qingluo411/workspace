# AGENTS.md — 自研 AI 加速芯片（DLC / TYD）软件栈速查

> 本文档用于快速回忆本仓库群（`/root/workspace`）的全貌：芯片、分层架构、
> 各仓库职责、构建顺序、运行时调用链与当前环境状态。
>
> 最后更新依据：仓库 HEAD 见下表；环境为 8 卡 TYD 真机 + 本地 LLVM 工具链。

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
| 驱动版本 | `Driver Version_v23.0.1`，`Api_Version_23` |
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

| 仓库 | 职责 | 产物 / 安装位置 | HEAD |
|---|---|---|---|
| **LLVM** | DLC TPU 编译器后端（`--target=dlc`，`-mcpu=dlc-gen2`，`-std=dc99`，注册 target 名 `dlc`） | `build/bin/{clang,ld.lld,llvm-mc,llvm-objcopy,elf2hex}` | `2fb2abd9` |
| **DLCsim** | 指令级模拟器 + **CRT 固件**（`lib-rt/crt0.S` 等） | `/usr/local/chipltech/simulator/lib/{libsimulator.so,libserver.so,dlc_crt*.hex,dlc_crt.rtabi}` | `5bfa8963` |
| **DLCSynapse** | Synapse 运行时核心（图、流、内存、kernel DB、profiler、用户态驱动） | `/usr/local/chipltech/synapse/lib/{libsynapse_core.so,libsynapse_backend.so,libdlc_runtime.so,libhl-thunk.so}` | `370fb6f8` |
| **DLC_Custom_Kernel** | 2686 个 AOT kernel（`dlc_kernels/*.{c,cpp}`） | `libcustom_dlc_perf_lib.so`（**约 1.4 GB**，kernel 以 blob 内嵌） | `b0d8998` |
| **DLC_CL** | 集合通信（DLCCL） | `/usr/local/chipltech/dlccl/lib/libdlccl.so` | `c4b2968` |
| **pytorch** | `device='dlc'` 后端 | editable 安装，`torch.dlc.*` | `6ab64d7a` |
| **vllm** / **vllm-cl** | 推理服务 + DLC hardware plugin | editable 安装 | `a0757668` / `d6e7a0b` |

> 所有仓库 remote 均为 `git@github.com:ChipLTech/<repo>.git`。
> 注意：`DLC_Custom_Kernel`/`pytorch`/`vllm` 当前在 `test-glm53flash` / `feat/glm5-flash-native-bringup` 等特性分支。

### 官方依赖图（`DLC_Custom_Kernel/README.md`）

```
LLVM ──┬─→ DLCsim ──┬─→ DLCSynapse ──┬─→ DLC_Custom_Kernel ──┐
       │            │               ├─→ DLC_CL              ├─→ pytorch ─→ vllm ─→ vllm-cl
       └────────────┴───────────────┘                       │
DLC-Kernel-Driver ─────────────────→ DLCSynapse              ┘
```

`/root/workspace/update.sh` 就是按此顺序串起来的一键构建脚本。

---

## 4. 构建顺序与产物布局

### 4.1 一键脚本 `update.sh`

```bash
cd LLVM            && ./build.sh
cd DLCsim          && ./build.sh
cd DLCSynapse      && ./compile.sh
cd DLC_Custom_Kernel && ./build.sh
cd DLC_CL          && ./build.sh
cd pytorch         && rm -rf build/* dist/* && USE_CUDA=0 python3 setup.py clean
                      PYTORCH_BUILD_VERSION=2.5.0 PYTORCH_BUILD_NUMBER=0 \
                      USE_CUDA=0 DEBUG=0 python3 setup.py develop
```

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
| `DLC_SKIP_VERSION_CHECK` | 跳过版本校验 |
| `DLC_SIM_ENHANCED_*` / `DLC_SIM_ENABLE_*` | 模拟器各类调试打印（见 `DLCsim/setenv.sh` 全表） |

> 注意：`.bashrc` 与 `/etc/environment` 不一致会导致交互/非交互行为不同。

---

## 7. 当前环境状态（2026-10 记录）

| 组件 | 源码构建 | 已安装 | 状态 |
|---|---|---|---|
| LLVM | ✅ | — | 完成 |
| DLCsim | ✅ | ✅ | 完成（hex 已是 `0TYD`；`lib-rt/*.hex` 在 git 中为 modified，未提交） |
| DLCSynapse | ✅ | ✅ | 完成 |
| DLC_Custom_Kernel | ✅ | ✅ | 完成（1.4GB `.so`） |
| DLC_CL | ✅ | ✅ | 完成 |
| pytorch | ⚠️ `build/lib/` 有 `libtorch_cpu.so`/`libtorch_dlc.so`/`libtorch_global_deps.so`，但 `torch/lib/` 空、无 `torch/_C.so` | — | **未收尾** |

- 当前 `import torch` 失败：
  ```
  OSError: /root/workspace/pytorch/torch/lib/libtorch_global_deps.so: cannot open shared object file
  ```
  原因：`setup.py develop` 编译已完成，但把 `build/lib/*.so` 落到 `torch/lib/` 并生成 `_C.so` 的收尾步骤未走完。
- 修复动作：重跑 `update.sh` 最后一段（`USE_CUDA=0 DEBUG=0 python3 setup.py develop`）。
- 真机 `device_init_flag` 为 `0x30000FF`（bit6/bit7 已置位 = CRT 已加载校验通过）。
- Python 环境：`/root/workspace/.venv`，Python 3.10.12，torch / vllm / vllm-cl 均为 editable 安装。

---

## 8. 常见问题排查入口

| 现象 | 先查 |
|---|---|
| `CRT TargetArch mismatch` | 第 5.2 节：`head -c 4` 检查 4 个 hex；确认 `DLC_TPU_VERSION` 与 `LLVM_PATH` |
| `import torch` 找不到 `.so` | `pytorch/torch/lib/` 是否为空 → 重跑 `setup.py develop` |
| 找不到 `libsimulator.so` / `libhl-thunk.so` | `LD_LIBRARY_PATH` 是否包含 `simulator/lib`、`synapse/lib` |
| 真机连不上 | `ls /dev/cltech*`、`lsmod \| grep chipltech`、`cat /sys/devices/virtual/chipltech/cltech0/version` |
| kernel 未生效/回退 CPU | `dlc_src/enabled_kernels.hpp` 的 `DispatchType`、`libcustom_dlc_perf_lib.so` 是否最新 |
| 想只用模拟器 | `export DLC_SYN_USE_SIM=1` |
| 性能/时序分析 | `synapse_perf/syn_perf` + `DLCsim/profiler` |

---

## 9. 目录速查（`/root/workspace`）

```
DLCSynapse/        运行时核心（synapse_core / synapse_backend / dlc_runtime / dlc_thunk / synapse_perf）
DLCsim/            模拟器 + CRT + server（lib-rt/、device/、src/xys、profiler/）
DLC_Custom_Kernel/ 2686 个 AOT kernel（dlc_kernels/）+ 分发元信息（dlc_src/）
DLC_CL/            集合通信 DLCCL
LLVM/              DLC 编译器工具链（dlc target）
pytorch/           PyTorch fork（device='dlc'）
vllm/              vLLM fork
vllm-cl/           vLLM DLC hardware plugin
update.sh          一键构建脚本
pyproject.toml     uv 管理的 workspace 元数据
```
