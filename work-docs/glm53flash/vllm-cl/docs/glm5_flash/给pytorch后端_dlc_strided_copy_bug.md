# [Bug] DLC 后端 strided→contiguous 拷贝在大张量 + 非 512B 对齐基址下静默算错

**优先级**：P0（静默数值错误，无任何报错，影响所有 strided 拷贝路径）
**组件**：pytorch DLC 后端（`aten/src/ATen/dlc`），torch-dlc / DLC 运行时
**版本**：`pytorch` 分支 `test-glm53flash`（HEAD `a18fc46ddcd`）；设备 `dlc:0/2/4/5` 结果一致
**复现脚本**：`docs/glm5_flash/op_requests/probes/probe_dlc_strided_copy_bug.py`
**复现命令**：`python -u docs/glm5_flash/op_requests/probes/probe_dlc_strided_copy_bug.py dlc:4`

## 一、问题

对**基地址不是 512B 对齐**的 strided 张量做大尺寸拷贝时，DLC 后端返回/写入**错误数据且不报错**：

```python
R, C, off = 3072, 12288, 1
big = torch.randn(R, C + off, dtype=torch.float32, device="dlc:0")
src = big[:, off:off + C]                       # stride=(C+1, 1)，基址 = +4B（非 512B 对齐）
got = src.contiguous()                          # 期望等于 big[:, 1:C+1]
err = (got.cpu() - big.cpu()[:, off:off + C]).abs().max()   # 实测 3.49（应为 0）
```

## 二、实测结果

| 用例（fp32，`R=3072, C=12288`） | 结果 |
|---|---|
| `src.contiguous()`，`off=1`（4B） | **err=3.49，错** |
| `src.clone()`，`off=1` | **err=3.49，错** |
| `torch.empty(R,C).copy_(src)`，`off=1` | **err=3.49，错** |
| 写方向：`big[:, 1:C+1].copy_(ones)` | **err=1.0，错**（静默写错值，未越界写） |
| `src.contiguous()` / `clone()` / `copy_()`，`off=128`（512B 对齐） | err=0，正确 |
| `src.to(memory_format=torch.contiguous_format)`，`off=1` | err=0，正确（可作临时规避） |
| 小尺寸：`R=3072, C=4096/8192/11264, off=1` | err=0，正确 |
| 尺寸阈值：`R=3072, C=12288, off=1` | **err=3.46，错** |
| 行数阈值：`C=12288, off=1`，`R=1024` / `R=2048` | 正确 / **错** |
| 3D 视图 `pool[7].copy_()`（偏移 129024B，512B 对齐） | err=0，正确 |

**错误形态**（确定性，与数据、随机种子无关；同输入重复执行结果完全一致）：
坏行从**第 128 行**开始，按 128 行粒度成段出现（坏行区间 `(128,129), (132,135), (138,141), …`，周期 6 行里坏 4 行）；
坏行内部数据整体**错位了 `off` 个元素**（例如 `off=1` 时行内某段等于参考值右移 1 个元素）。
表现为：拷贝的首块正确，后续块沿用了错误的起始地址。
**触发条件**：strided 张量的**基址非 512B 对齐**（4B/8B/…/508B/1020B 都触发，128 个 fp32 即 512B 起正常）
**且**张量足够大（fp32 实测在 ~132MB→144MB、行数 1024→2048 之间出现）。

## 三、影响面

- 所有走 `at::native` strided 拷贝的实现都中招：`Tensor.contiguous()`、`Tensor.clone()`、`Tensor.copy_()`（读、写两个方向）。
- **DLC kernel launch 的自动兜底路径同样中招**：非连续入参经 `at::detail::to_contiguous_dlc`
  （`aten/src/ATen/dlc/DLCKernelLaunch.h:1052-1063`）拷贝、输出经 `post_fn` 里 `tensor.copy_(new_tensor)` 回写；
  即**任何接收大尺寸非连续 tensor 的 DLC kernel 都可能拿到被污染的数据**，且无任何告警。
- 规避（临时）：非 512B 对齐的大张量改走 `tensor.to(memory_format=torch.contiguous_format)`（实测正确），
  或先做 512B 对齐的视图/分块拷贝。

**我方影响评估（GLM-5.3-Flash）**：当前不受影响。mamba page = `18432B(conv) + 524288B(rec) = 542720B = 1060×512B`，
页内偏移 0 / 18432 均为 512B 倍数，按 block 取视图基址天然对齐、单块最大 1.5MB；KV cache、权重同理。
唯一命中的是验证脚本里的参考实现（`w[:,k:k+1] * xs[:,k:k+N]`，k=1..3 → 4/8/12B 偏移），
已改为 CPU 分块计算。**若后续插件在 torch 层对大张量做元素级切片后 `.contiguous()`，会静默算错。**

## 四、期望行为

上述所有用例结果与 CPU 路径**逐元素一致**（`err == 0`）；对齐与非对齐、大尺寸与小尺寸行为统一。
另外建议：对无法保证对齐的 strided 拷贝路径给出显式告警或回退到安全的逐元素实现，避免静默错误。

## 五、验收标准

- [ ] `probe_dlc_strided_copy_bug.py` 在 `dlc:0/2/4/5` 上全部用例 `err == 0`（当前脚本以 FAIL 退出）
- [ ] 覆盖 `contiguous` / `clone` / `copy_` 读写两方向、fp32/bf16/int32、多种 shape 与非对齐偏移
- [ ] 已将该场景加入 pytorch DLC 后端回归测试
