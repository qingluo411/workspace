#!/usr/bin/env python3
"""Attribute the device-idle gaps of one model step to host-side Python calls.

The decode step's device stream is idle ~65% of the time; that idle is not one
long stall but ~3900 inter-launch holes of ~100 us, plus a handful of multi-ms
stalls.  This script lists, per rank:

  * total blocked host time in `.tolist()` / `.cpu()` / `.numpy()` (blocking
    device->host reads), i.e. the calls that force the stream to drain;
  * how the two line up: the matching long `driver.memcpy` on stream:0 in the
    DLCSynapse device trace shows the cost is the D2H copy itself, not the CPU.

Usage: host_stall.py <rank0.pt.trace.json.gz> <syn_trace_dir> <out.txt>
"""
import glob
import gzip
import json
import os
import re
import sys

BLOCKING = ("tolist", "cpu of Tensor", "numpy of Tensor", "item of Tensor")
GHZ = 2.92
CYCLES = re.compile(r"crt0_cycles:(\d+)\s+xys0_cycles:(\d+)\s+crt1_cycles:(\d+)\s+xys1_cycles:(\d+)")


def main(torch_trace, syn_dir, out_path):
    with gzip.open(torch_trace, "rt") as handle:
        events = json.load(handle)["traceEvents"]
    steps = sorted(
        (e for e in events if e.get("cat") == "user_annotation"
         and e.get("name", "").startswith("execute_context") and (e.get("dur") or 0) > 1e5),
        key=lambda e: e["ts"],
    )
    step_index = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    step = steps[min(step_index, len(steps) - 1)]
    start, end = step["ts"], step["ts"] + step["dur"]
    pid0 = events[0]["pid"]

    lines = [f"step {step['name']} span={(end - start) / 1000:.1f} ms"]
    lines.append("\n== blocking host->device reads per rank ==")
    lines.append(f"{'rank':>4} {'tolist_n':>9} {'tolist_ms':>10} {'max_ms':>8} "
                 f"{'d2h_n':>6} {'d2h_ms':>8} {'d2h_max_ms':>11}")
    for rank in range(8):
        trace = sorted(glob.glob(f"{os.path.dirname(torch_trace)}/dp0_pp0_tp{rank}_dcp0_ep{rank}_rank*.pt.trace.json.gz"))[-1]
        with gzip.open(trace, "rt") as handle:
            ev = json.load(handle)["traceEvents"]
        rank_pid = ev[0]["pid"]
        tls = [e["dur"] for e in ev if e.get("ph") == "X" and e.get("cat") == "python_function"
               and any(k in (e.get("name") or "") for k in BLOCKING)
               and e.get("ts") is not None and start <= e["ts"] <= end]
        d2h = []
        with open(f"{syn_dir}/syn_trace_{rank_pid}.json") as handle:
            for line in handle:
                if "driver.memcpy" not in line:
                    continue
                try:
                    x = json.loads(line.rstrip().rstrip(","))
                except ValueError:
                    continue
                if x.get("pid") != "stream:0" or x.get("ts") is None or not (start <= x["ts"] <= end):
                    continue
                if (x.get("dur") or 0) > 500:
                    d2h.append(x["dur"])
        lines.append(f"{rank:>4} {len(tls):>9} {sum(tls) / 1000:>10.1f} "
                     f"{(max(tls) if tls else 0) / 1000:>8.1f} {len(d2h):>6} "
                     f"{sum(d2h) / 1000:>8.1f} {(max(d2h) if d2h else 0) / 1000:>11.1f}")

    lines.append("\n== inter-launch holes (device-idle distribution of rank0) ==")
    ks = []
    with open(f"{syn_dir}/syn_trace_{pid0}.json") as handle:
        for line in handle:
            if '"type:' not in line:
                continue
            try:
                x = json.loads(line.rstrip().rstrip(","))
            except ValueError:
                continue
            t = x.get("ts")
            if t is None or not (start <= t <= end):
                continue
            m = CYCLES.search((x.get("args") or {}).get("message", ""))
            if m and sum(int(g) for g in m.groups()) > 0:
                ks.append((t, t + (x.get("dur") or 0)))
    ks.sort()
    holes = [ks[i + 1][0] - ks[i][1] for i in range(len(ks) - 1) if ks[i + 1][0] > ks[i][1]]
    holes.sort(reverse=True)
    lines.append(f"kernels={len(ks)} holes={len(holes)} total_idle={sum(holes) / 1000:.1f} ms "
                 f"mean_hole={sum(holes) / len(holes) / 1000:.3f} ms max_hole={holes[0] / 1000:.3f} ms")
    lines.append("largest holes (ms): " + ", ".join(f"{h / 1000:.3f}" for h in holes[:15]))

    text = "\n".join(lines) + "\n"
    open(out_path, "w").write(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
