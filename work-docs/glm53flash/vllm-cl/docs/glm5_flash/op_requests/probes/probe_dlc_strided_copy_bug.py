"""最小复现：DLC(pytorch) 后端 strided->contiguous 拷贝在大张量 + 非 512B 对齐基址下算错。

用法：
    python -u probe_dlc_strided_copy_bug.py dlc:0

结论（2026-09-28，在 dlc:0/2/4/5 上一致复现，确定性）：
  - `src.contiguous()` / `src.clone()` / `dst.copy_(src)` 三者都会算错；
  - 触发条件：strided 张量的**基地址不是 512B 对齐**，且张量足够大
    （实测 fp32 [3072, 12288] 出错、[3072, 8192] 及更小正确）；
  - 基址 512B 对齐（如列偏移 128 个 fp32）时结果完全正确；
  - `.to(memory_format=torch.contiguous_format)` 与 CPU 路径在同样输入下正确；
  - 错误是整块错位：坏行从第 128 行开始、按 128 行粒度成段出现。
"""

import sys

import torch

dev = sys.argv[1] if len(sys.argv) > 1 else "dlc:0"
torch.dlc.set_device(int(dev.split(":")[-1]))
torch.manual_seed(0)

R, C = 3072, 12288


def report(tag, got, ref):
    err = (got.cpu().float() - ref).abs().max().item()
    print(f"{'BAD ' if err > 1e-2 else 'ok  '} {tag:34s} err={err:.3e}", flush=True)
    return err


bad = 0

# ---- 1) 读方向：非 512B 对齐的切片 .contiguous() / .clone() / copy_ ----
for off, note in ((1, "4B (未对齐)"), (128, "512B (对齐)")):
    big = (torch.randn(R, C + off, dtype=torch.float32) * 0.5).to(dev)
    ref = big.cpu()[:, off : off + C]
    src = big[:, off : off + C]
    bad += report(f"contiguous() off={off} {note}", src.contiguous(), ref)
    bad += report(f"clone()      off={off} {note}", src.clone(), ref)
    dst = torch.empty(R, C, dtype=torch.float32, device=dev)
    dst.copy_(src)
    bad += report(f"copy_()      off={off} {note}", dst, ref)
    # 规避手段
    report(f"to(contiguous_format) off={off}", src.to(memory_format=torch.contiguous_format), ref)
    del big, src, ref, dst

# ---- 2) 写方向：往非 512B 对齐的 strided view 里 copy_ ----
for off, note in ((1, "4B (未对齐)"), (128, "512B (对齐)")):
    big = torch.zeros(R, C + off, dtype=torch.float32, device=dev)
    ones = torch.ones(R, C, dtype=torch.float32, device=dev)
    big[:, off : off + C].copy_(ones)
    bad += report(f"write into view off={off} {note}", big[:, off : off + C], torch.ones(R, C))

# ---- 3) 大小阈值：同样未对齐，小张量应正确 ----
for c in (4096, 8192, 11264, 12288):
    big = (torch.randn(R, c + 1, dtype=torch.float32) * 0.5).to(dev)
    report(f"size sweep R={R} C={c} off=1", big[:, 1 : 1 + c].contiguous(), big.cpu()[:, 1 : 1 + c])
    del big

print(f"\n{'FAIL: 复现了 DLC strided 拷贝缺陷' if bad else 'PASS: 未复现'}", flush=True)
sys.exit(1 if bad else 0)
