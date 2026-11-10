#!/usr/bin/env python3
"""Per-operator input shapes (+ device cycles) for GLM-5.3-Flash on DLC.

Source is the Synapse kernel-detail log written when the service runs with
DLC_SYN_LOG_DIR / DLC_SYN_VERBOSE=1 / DLC_SYN_DEBUG=1 / DLC_SYN_PROF_CYCLE=1
(see start_glm5_flash_vllm_cl.sh --kernel-log).  Each kernel launch produces:

  ───── launch  <name>  on dlc:7 (8 'DLC') stream=0x... nbytes=NNN
  DLCTensor[1]:
     0: addr=0x...  shape=[...]  stride=[...]  offset=0  dim0,1=... padding_bytes=... dtype=DLCBF16 ...
  DLCScalar[1]:
     0: value = 1e-05  dtype = DLCFP32
  ───── cycles: 2458  ops: 1  bytes: 4  crt: 0  gflops: ... gb/s: ...

Usage:
  op_shapes_from_kernel_log.py <syn_<pid>.ansi> [--offset N] [--out-dir D]
Window: --offset is the file size recorded just before the request (logs are
append-only), so the parse covers exactly that request.
Phase split: everything before the first `custom_dlc_kda_recurrent_decode`
launch is reported as prefill, from there on as decode.  The boundary lands in
the prefill sampler / decode input-prep region.
"""
import argparse
import collections
import os
import re

HDR = re.compile(r"^─+ launch  (\S+)  on dlc:(\d+) .*nbytes=(\d+)\s*$")
FTR = re.compile(r"^─+ cycles: (\d+)  ops: (\d+)  bytes: (\d+)  crt: (\d+)")
TEN = re.compile(
    r"shape=\[([^\]]*)\]\s+stride=\[([^\]]*)\]\s+offset=(\d+)\s+dim0,1=[\d,]*"
    r"\s+padding_bytes=(\d+)\s+dtype=(\w+)"
)
SPLIT_KERNEL = "custom_dlc_kda_recurrent_decode"


def parse_blocks(path, offset=0):
    blocks = []
    name = None
    body = []
    with open(path, errors="ignore") as handle:
        handle.seek(offset)
        for line in handle:
            header = HDR.match(line)
            if header:
                if name is not None:
                    blocks.append((name, None, "".join(body)))
                name, body = header.group(1), [line]
                continue
            if name is None:
                continue
            body.append(line)
            footer = FTR.match(line)
            if footer:
                blocks.append((name, int(footer.group(1)), "".join(body)))
                name, body = None, []
    if name is not None:
        blocks.append((name, None, "".join(body)))
    return [block for block in blocks if block[1] is not None]


def shape_signature(body):
    tensors = TEN.findall(body)
    if not tensors:
        return "<no DLCTensor>"
    return " ".join(
        f"[{shape}]/{dtype.replace('DLC', '').lower()}"
        for shape, _, _, _, dtype in tensors
    )


def report(tag, blocks, out):
    total = sum(cycles for _, cycles, _ in blocks)
    per_op = collections.defaultdict(lambda: [0, 0])
    signatures = collections.defaultdict(list)
    for name, cycles, body in blocks:
        row = per_op[name]
        row[0] += 1
        row[1] += cycles
        signature = shape_signature(body)
        if signature not in signatures[name]:
            signatures[name].append(signature)

    lines = [
        f"# {tag}: blocks={len(blocks)} total_cycles={total:,} ops={len(per_op)}",
        f"# {'calls':>7s} {'cycles':>15s} {'share':>7s}  operator",
    ]
    for name, (calls, cycles) in sorted(per_op.items(), key=lambda kv: -kv[1][1]):
        lines.append(f"  {calls:7d} {cycles:15,d} {100.0 * cycles / total:6.2f}%  {name}")
        for signature in signatures[name]:
            lines.append(f"          {signature}")
    text = "\n".join(lines) + "\n"
    out.write(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    blocks = parse_blocks(args.log, args.offset)
    cut = next(
        (index for index, (name, _, _) in enumerate(blocks) if name == SPLIT_KERNEL),
        len(blocks),
    )
    os.makedirs(args.out_dir, exist_ok=True)
    for tag, segment in (("prefill", blocks[:cut]), ("decode", blocks[cut:])):
        with open(os.path.join(args.out_dir, f"op_shapes_{tag}.txt"), "w") as out:
            text = report(tag, segment, out)
        first = text.splitlines()[0]
        print(f"{first} -> {os.path.join(args.out_dir, f'op_shapes_{tag}.txt')}")


if __name__ == "__main__":
    main()
