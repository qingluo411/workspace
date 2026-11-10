#!/usr/bin/env python3
"""Per-rank split of the pre-collective sync kernel vs the ring-allreduce body.

Both kernels live on the rank's main compute stream, so their device `dur`
values never overlap and can simply be summed (verified: sum(dur) == union(dur)
over every kernel in the window).  The DLCSynapse `xys*/crt*_cycles` counters are
divided by the 2.92 GHz XYS clock, which is what makes cycles->ms consistent
with the trace's own `dur` field.

Usage: python rank_sweep.py <rank0.pt.trace.json.gz> <syn_trace_dir>
"""
import glob
import gzip
import json
import os
import re
import sys

CYCLES = re.compile(
    r"crt0_cycles:(\d+)\s+xys0_cycles:(\d+)\s+crt1_cycles:(\d+)\s+xys1_cycles:(\d+)"
)
GHZ = 2.92
SYNC = "custom_sync_all_machine"
RINGS = (
    "custom_allreduce_ring_simple_bf16_rdma",
    "custom_allreduce_2x4tpu1root_simple_bf16_rdma",
)


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


def windows(torch_trace):
    with gzip.open(torch_trace, "rt") as handle:
        events = [e for e in json.load(handle)["traceEvents"] if e.get("ph") == "X"]
    steps = sorted(
        (e for e in events if e.get("cat") == "user_annotation"
         and e.get("name", "").startswith("execute_context") and (e.get("dur") or 0) > 1e5),
        key=lambda e: e["ts"],
    )
    return [(f"prefill" if i == 0 else f"decode{i}", s["ts"], s["ts"] + s["dur"])
            for i, s in enumerate(steps)]


def main(torch_trace, syn_dir):
    windows_ = windows(torch_trace)
    files = rank_syn_traces(torch_trace, syn_dir)
    rows = []
    decode_sync = {}
    for rank, path in enumerate(files):
        pid = int(re.search(r"syn_trace_(\d+)", path).group(1))
        acc = {name: {"sync": [0, 0], "ring": [0, 0]} for name, _, _ in windows_}
        with open(path) as handle:
            for line in handle:
                if SYNC not in line and not any(ring in line for ring in RINGS):
                    continue
                try:
                    event = json.loads(line.rstrip().rstrip(","))
                except ValueError:
                    continue
                stamp = event.get("ts")
                match = CYCLES.search((event.get("args") or {}).get("message", ""))
                if stamp is None or not match:
                    continue
                cycles = sum(int(g) for g in match.groups())
                key = "sync" if event["name"] == SYNC else "ring"
                for name, start, end in windows_:
                    if start <= stamp <= end:
                        row = acc[name][key]
                        row[0] += 1
                        row[1] += cycles
                        break
        for name, _, _ in windows_:
            sync, ring = acc[name]["sync"], acc[name]["ring"]
            if not sync[0] and not ring[0]:
                continue
            rows.append((rank, name, sync[0], sync[1], ring[0], ring[1]))
            if name.startswith("decode"):
                decode_sync[rank] = decode_sync.get(rank, 0) + sync[1]

    root_rank = min(decode_sync, key=decode_sync.get) if decode_sync else None
    print(f"root_rank={root_rank}: its sync kernel releases the barrier, so sync_us is not wait time.")
    print(f"{'rank':>4} {'role':>5} {'phase':>8} {'n':>4} {'sync_ms':>8} {'sync_us':>8} "
          f"{'ring_ms':>8} {'ring_us':>8} {'sync/ring':>9}")
    for rank, name, sync_n, sync_cycles, ring_n, ring_cycles in rows:
        role = "root" if rank == root_rank else ""
        print(f"{rank:>4} {role:>5} {name:>8} {sync_n:>4} "
              f"{sync_cycles / GHZ / 1e6:>8.1f} {sync_cycles / max(sync_n, 1) / GHZ / 1e3:>8.1f} "
              f"{ring_cycles / GHZ / 1e6:>8.1f} {ring_cycles / max(ring_n, 1) / GHZ / 1e3:>8.1f} "
              f"{sync_cycles / max(ring_cycles, 1):>9.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
