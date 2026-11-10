#!/usr/bin/env python3
"""真机验证 C1：`MetadataPacker` 的段视图在 DLC launcher 眼里是否仍「免拷贝」。

判据见 `aten/src/ATen/dlc/DLCKernelLaunch.h:1046` / `DLCTensorImpl.h:56`，
设计约束见 `docs/glm5_flash/元数据H2D归因与合并方案_2026-10-09.md` §3.2.1：

  * 段起点按 align 元素对齐、段容量按 align 元素取整后，任意「align 整数倍长度」
    的切片与独立分配时一样不触发 launcher 的自动连续化拷贝；
  * 长度非 align 整数倍的切片本来就会拷贝（打包前后一致）；
  * `query_start_loc`（max_num_reqs + 1 = 257）靠 exact-fit 规则免拷贝，
    所以必须独立成桶，不能塞进大 buffer。

用法：
    cd /root/workspace/vllm-cl
    /root/miniconda/envs/qingluo-dev1/bin/python -u \
        docs/glm5_flash/probes/probe_metadata_packer_layout.py dlc:0
"""

from __future__ import annotations

import statistics
import sys
import time

import torch

from vllm_cl.worker.metadata_packer import MetadataPacker

MAX_NUM_TOKENS = 8192
MAX_NUM_REQS = 256


def bench(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.dlc.synchronize()
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.dlc.synchronize()
        samples.append((time.perf_counter() - t0) * 1e6 / iters)
    return statistics.median(samples)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dev = args[0] if args else "dlc:0"
    torch.dlc.set_device(int(dev.split(":")[-1]))
    device = torch.device(dev)
    print(f"device={dev}", flush=True)

    packer = MetadataPacker(device=device, pin_memory=True)
    packed = {
        "input_ids": packer.add("input_ids", MAX_NUM_TOKENS, torch.int32),
        "num_scheduled_tokens": packer.add(
            "num_scheduled_tokens", MAX_NUM_REQS, torch.int32
        ),
        "req_indices": packer.add("req_indices", MAX_NUM_TOKENS, torch.int64),
        "query_pos": packer.add("query_pos", MAX_NUM_TOKENS, torch.int64),
    }
    packer.finalize()
    print("bucket layout:")
    for key, spec in packer.bucket_layout().items():
        print(f"  {key}: {spec}")

    # 独立分配的对照组（与打包前一致）。
    standalone = {
        "input_ids": torch.zeros(MAX_NUM_TOKENS, dtype=torch.int32, device=device),
        "num_scheduled_tokens": torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device=device),
        "req_indices": torch.zeros(MAX_NUM_TOKENS, dtype=torch.int64, device=device),
        "query_pos": torch.zeros(MAX_NUM_TOKENS, dtype=torch.int64, device=device),
    }
    standalone["query_start_loc"] = torch.zeros(
        MAX_NUM_REQS + 1, dtype=torch.int32, device=device
    )

    print(f"\n{'case':44} {'packed us':>10} {'solo us':>10} {'ratio':>7}")
    for name in ("input_ids", "req_indices", "query_pos"):
        align = 128
        for n in (align, align * 16, align * 16 + 1):
            seg = packed[name]
            solo = standalone[name]
            packed_us = bench(lambda s=seg, k=n: torch.add(s.gpu[:k], s.gpu[:k]))
            solo_us = bench(lambda t=solo, k=n: torch.add(t[:k], t[:k]))
            print(
                f"{name + f'[:{n}]':44} {packed_us:10.2f} {solo_us:10.2f} "
                f"{packed_us / solo_us:7.2f}",
                flush=True,
            )

    for n in (128, MAX_NUM_REQS):
        seg = packed["num_scheduled_tokens"]
        solo = standalone["num_scheduled_tokens"]
        packed_us = bench(lambda s=seg, k=n: torch.add(s.gpu[:k], s.gpu[:k]))
        solo_us = bench(lambda t=solo, k=n: torch.add(t[:k], t[:k]))
        print(
            f"{'num_scheduled_tokens' + f'[:{n}]':44} {packed_us:10.2f} "
            f"{solo_us:10.2f} {packed_us / solo_us:7.2f}",
            flush=True,
        )

    # exact-fit 规则：257 的独立 buffer 免拷贝，塞进大桶就会退化。
    qsl = standalone["query_start_loc"]
    big = torch.zeros(MAX_NUM_TOKENS + MAX_NUM_REQS, dtype=torch.int32, device=device)
    qsl_packed_bad = big.narrow(0, 128, MAX_NUM_REQS + 1)
    a = bench(lambda t=qsl: torch.add(t[: MAX_NUM_REQS + 1], t[: MAX_NUM_REQS + 1]))
    b = bench(
        lambda t=qsl_packed_bad: torch.add(
            t[: MAX_NUM_REQS + 1], t[: MAX_NUM_REQS + 1]
        )
    )
    print(
        f"\nquery_start_loc[:257] standalone={a:.2f}us "
        f"in-big-buffer={b:.2f}us ratio={b / a:.2f} "
        f"(=> 必须 isolated，设计见 §3.2.1)",
        flush=True,
    )

    # 端到端 H2D 次数：打包后应有 4 次（int32 桶 + int64 桶 + 2 个独立 buffer）。
    packer.mark("input_ids", MAX_NUM_TOKENS)
    packer.mark("num_scheduled_tokens", MAX_NUM_REQS)
    packer.mark("req_indices", MAX_NUM_TOKENS)
    packer.mark("query_pos", MAX_NUM_TOKENS)
    packer.flush()
    torch.dlc.synchronize()
    print("flush() done", flush=True)


if __name__ == "__main__":
    main()
