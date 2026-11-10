#!/usr/bin/env python3
"""Per-operator device-cycle share for GLM-5.3-Flash on DLC.

Two inputs, same µs clock (the torch trace and the runtime trace both use it):
  1. a vLLM worker torch trace (*.pt.trace.json[.gz]) -- used only to find the
     model-step windows (execute_context_* user annotations);
  2. a DLCSynapse runtime trace (syn_trace_<pid>.json, written when the service
     runs with DLC_SYN_PROF_DIR set) -- carries the per-kernel device cycles in
     the `args.message` of the `type:*` device lanes:
         idx:0 of 1  crt0_cycles:0  xys0_cycles:2458  crt1_cycles:0  xys1_cycles:2351

Usage:
  python op_share_cycles.py <worker.pt.trace.json.gz> <syn_trace.json> [out_dir]
Writes <out_dir>/op_share_<step>.txt (count, cycles, pct, cum_pct, name).
"""
import collections
import gzip
import json
import os
import re
import sys

CYCLE_RE = re.compile(
    r"crt0_cycles:(\d+)\s+xys0_cycles:(\d+)\s+crt1_cycles:(\d+)\s+xys1_cycles:(\d+)"
)


def load_events(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        return json.load(handle)["traceEvents"]


def step_windows(trace_path):
    events = [e for e in load_events(trace_path) if e.get("ph") == "X"]
    steps = sorted(
        (
            e
            for e in events
            if e.get("cat") == "user_annotation"
            and e.get("name", "").startswith("execute_context")
            and (e.get("dur") or 0) > 1e5
        ),
        key=lambda e: e["ts"],
    )
    # The torch annotation covers the step; pad by the previous step so ops that
    # land just outside the annotation on another stream still count.
    windows = []
    for index, step in enumerate(steps):
        start = step["ts"]
        end = start + step["dur"]
        windows.append((f"step{index}_{step['name']}", start, end))
    return windows


def aggregate(syn_trace_path, windows):
    agg = {name: collections.defaultdict(lambda: [0, 0]) for name, _, _ in windows}
    with open(syn_trace_path) as handle:
        for line in handle:
            if '"type:' not in line:
                continue
            try:
                event = json.loads(line.rstrip().rstrip(","))
            except ValueError:
                continue
            ts = event.get("ts")
            if ts is None:
                continue
            match = CYCLE_RE.search((event.get("args") or {}).get("message", ""))
            if not match:
                continue
            cycles = sum(int(group) for group in match.groups())
            for name, start, end in windows:
                if start <= ts <= end:
                    row = agg[name][event["name"]]
                    row[0] += 1
                    row[1] += cycles
    return agg


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    trace_path, syn_trace_path = sys.argv[1], sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "."
    windows = step_windows(trace_path)
    if not windows:
        raise SystemExit("no model step found in the torch trace")
    agg = aggregate(syn_trace_path, windows)
    os.makedirs(out_dir, exist_ok=True)
    for name, _, _ in windows:
        rows = sorted(agg[name].items(), key=lambda kv: -kv[1][1])
        total = sum(v[1] for v in agg[name].values())
        path = os.path.join(out_dir, f"op_share_{name}.txt")
        with open(path, "w") as out:
            cumulative = 0
            for op, (calls, cycles) in rows:
                cumulative += cycles
                out.write(
                    f"{calls:7d} {cycles:>14,} {100.0 * cycles / total:6.2f}% "
                    f"{100.0 * cumulative / total:6.2f}%  {op}\n"
                )
        print(f"{name}: ops={len(rows)} kernels={sum(v[0] for v in agg[name].values())} "
              f"total_cycles={total:,} -> {path}")


if __name__ == "__main__":
    main()
