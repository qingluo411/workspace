#!/usr/bin/env python3
"""What is custom_sync_all_machine waiting for?  Cross-rank barrier analysis.

For every barrier (= every custom_sync_all_machine instance) in one model step we
compare the 8 ranks' arrival and release timestamps:

  arrival = sync kernel start ts          (when that rank reaches the barrier)
  release = sync kernel start ts + dur    (when the barrier lets everybody go)

A barrier whose releases coincide (spread ~0) but whose arrivals are spread apart
is pure rendezvous waiting, and the spread is the inter-rank skew of the work
that precedes it.  The script then decomposes the skewed segments into
device-busy vs device-idle, per rank, to show whether the skew is compute
imbalance or host-side jitter.

Usage: barrier_wait.py <rank0.pt.trace.json.gz> <syn_trace_dir> <out.txt>
The step window comes from rank0's `execute_context_*` annotation; the DLCSynapse
device trace shares that µs clock.  Worker PIDs are not guaranteed to be
contiguous, so each rank's torch trace is used to resolve its own
syn_trace_<pid>.json file.
"""
import gzip
import glob
import json
import os
import re
import statistics
import sys

CYCLES = re.compile(r"crt0_cycles:(\d+)\s+xys0_cycles:(\d+)\s+crt1_cycles:(\d+)\s+xys1_cycles:(\d+)")
SYNC = "custom_sync_all_machine"
RING = "custom_allreduce_ring_simple_bf16_rdma"


def union(intervals):
    total = 0
    cur_s = cur_e = None
    for start, end in sorted(intervals):
        if cur_s is None:
            cur_s, cur_e = start, end
        elif start <= cur_e:
            cur_e = max(cur_e, end)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = start, end
    if cur_s is not None:
        total += cur_e - cur_s
    return total


def step_window(trace_path):
    with gzip.open(trace_path, "rt") as handle:
        events = [e for e in json.load(handle)["traceEvents"] if e.get("ph") == "X"]
    steps = sorted(
        (e for e in events if e.get("cat") == "user_annotation"
         and e.get("name", "").startswith("execute_context") and (e.get("dur") or 0) > 1e5),
        key=lambda e: e["ts"],
    )
    return steps


def rank_syn_traces(torch_trace, syn_dir):
    """Resolve eight syn traces from the per-rank torch trace worker PIDs."""
    trace_dir = os.path.dirname(os.path.abspath(torch_trace))
    reference_mtime = os.path.getmtime(torch_trace)
    torch_traces = {}
    for path in glob.glob(
        os.path.join(trace_dir, "dp0_pp0_tp*_dcp0_ep*_rank*.pt.trace.json.gz")
    ):
        match = re.search(r"_tp(\d+)_dcp0_ep\d+_rank\d+\.", os.path.basename(path))
        if match:
            torch_traces.setdefault(int(match.group(1)), []).append(path)

    traces = []
    for rank in range(8):
        candidates = [
            path for path in torch_traces.get(rank, [])
            if abs(os.path.getmtime(path) - reference_mtime) <= 600
        ]
        if not candidates:
            raise FileNotFoundError(
                f"missing torch trace for rank {rank} near {torch_trace}"
            )
        rank_trace = (
            torch_trace if rank == 0
            else min(candidates, key=lambda path: abs(os.path.getmtime(path) - reference_mtime))
        )
        with gzip.open(rank_trace, "rt") as handle:
            events = json.load(handle)["traceEvents"]
        worker_pid = next(event["pid"] for event in events if event.get("pid") is not None)
        traces.append(os.path.join(syn_dir, f"syn_trace_{worker_pid}.json"))

    missing = [path for path in traces if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"missing syn trace(s): {', '.join(missing)}")
    return traces


def main(torch_trace, syn_dir, out_path):
    steps = step_window(torch_trace)
    step_index = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    step = steps[min(step_index, len(steps) - 1)]       # decode
    start, end = step["ts"], step["ts"] + step["dur"]

    traces = rank_syn_traces(torch_trace, syn_dir)
    kernels, syncs = {}, {}
    for rank, path in enumerate(traces):
        ks, sy = [], []
        with open(path) as handle:
            for line in handle:
                if '"type:' not in line:
                    continue
                try:
                    event = json.loads(line.rstrip().rstrip(","))
                except ValueError:
                    continue
                stamp = event.get("ts")
                if stamp is None or not (start <= stamp <= end):
                    continue
                match = CYCLES.search((event.get("args") or {}).get("message", ""))
                if not match or sum(int(g) for g in match.groups()) == 0:
                    continue
                ks.append((stamp, event.get("dur") or 0, event["name"]))
                if event["name"] == SYNC:
                    sy.append((stamp, event.get("dur") or 0))
        kernels[rank], syncs[rank] = sorted(ks), sorted(sy)

    ranks_with_sync = [r for r in range(8) if syncs[r]]
    root_rank = (
        min(ranks_with_sync, key=lambda r: sum(duration for _, duration in syncs[r]))
        if ranks_with_sync else None
    )
    lines = [
        f"step {step['name']} span={(end - start) / 1000:.1f} ms  "
        f"barriers/rank={[len(syncs[r]) for r in range(8)]}  root_rank={root_rank}"
    ]
    lines.append(
        "root rank releases the barrier, so its sync duration is near zero; "
        "its arrival time is still on the critical path."
    )

    lines.append("\n== per-barrier arrival / release (ms) ==")
    lines.append(f"{'idx':>4} {'arrive_spread':>13} {'release_spread':>14} {'last_rank':>9} {'wait_max':>9}")
    spreads = []
    for i in range(min(len(syncs[r]) for r in range(8))):
        arrive = [syncs[r][i][0] for r in range(8)]
        release = [syncs[r][i][0] + syncs[r][i][1] for r in range(8)]
        spread = (max(arrive) - min(arrive)) / 1000
        spreads.append((spread, i, arrive.index(max(arrive))))
    spreads.sort(reverse=True)
    for spread, i, last in spreads[:12]:
        arrive = [syncs[r][i][0] for r in range(8)]
        release = [syncs[r][i][0] + syncs[r][i][1] for r in range(8)]
        waits = [(syncs[r][i][0] + syncs[r][i][1] - syncs[r][i][0]) / 1000 for r in range(8)]
        lines.append(f"{i:>4} {spread:>13.3f} {(max(release) - min(release)) / 1000:>14.3f} "
                     f"{'rank' + str(last):>9} {max(waits):>9.3f}")
    allsp = [s for s, _, _ in spreads]
    lines.append("arrival spread over all barriers: mean %.3f ms, median %.3f ms, max %.3f ms, sum %.1f ms"
                 % (statistics.mean(allsp), statistics.median(allsp), max(allsp), sum(allsp)))
    relsp = [(max(syncs[r][i][0] + syncs[r][i][1] for r in range(8))
              - min(syncs[r][i][0] + syncs[r][i][1] for r in range(8))) / 1000
             for i in range(min(len(syncs[r]) for r in range(8)))]
    lines.append("release spread over all barriers: mean %.3f ms, max %.3f ms  <- the barrier itself is exact"
                 % (statistics.mean(relsp), max(relsp)))

    lines.append("\n== top skewed segments: is the laggard doing more work, or just slower to launch? ==")
    for spread, i, last in [s for s in spreads if s[1] > 0][:3]:
        lines.append(f"\n--- segment {i - 1} -> {i}, arrival spread {spread:.3f} ms, laggard rank{last}")
        lines.append(f"{'rank':>4} {'segment_ms':>11} {'kernels':>8} {'busy_ms':>8} {'idle_ms':>8}")
        for r in range(8):
            seg_start = syncs[r][i - 1][0] + syncs[r][i - 1][1]
            seg_end = syncs[r][i][0]
            ks = [(t, t + d) for t, d, _ in kernels[r] if seg_start <= t < seg_end]
            busy = union(ks)
            lines.append(f"{r:>4} {(seg_end - seg_start) / 1000:>11.3f} {len(ks):>8} "
                         f"{busy / 1000:>8.3f} {(seg_end - seg_start - busy) / 1000:>8.3f}")

    lines.append("\n== per-rank step totals ==")
    lines.append(f"{'rank':>4} {'role':>5} {'kernels':>8} {'busy_ms':>8} {'sync_ms':>8} {'useful_ms':>10} {'idle_ms':>8} {'span_ms':>8}")
    for r in range(8):
        busy = sum(d for _, d, _ in kernels[r])
        sync = sum(d for _, d in syncs[r])
        role = "root" if r == root_rank else ""
        lines.append(f"{r:>4} {role:>5} {len(kernels[r]):>8} {busy / 1000:>8.1f} {sync / 1000:>8.1f} "
                     f"{(busy - sync) / 1000:>10.1f} {(end - start - busy) / 1000:>8.1f} {(end - start) / 1000:>8.1f}")

    text = "\n".join(lines) + "\n"
    open(out_path, "w").write(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
