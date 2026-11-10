#!/usr/bin/env python3
"""Window analysis: cycles share (sum) vs wall-clock occupancy (interval union),
split by stream lane, for the GLM-5.3-Flash allreduce/sync attribution."""
import collections, gzip, json, re, sys

CY = re.compile(r"crt0_cycles:(\d+)\s+xys0_cycles:(\d+)\s+crt1_cycles:(\d+)\s+xys1_cycles:(\d+)")

def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as h:
        return json.load(h)["traceEvents"]

def windows(tp):
    ev = [e for e in load(tp) if e.get("ph") == "X"]
    steps = sorted((e for e in ev if e.get("cat") == "user_annotation"
                    and e.get("name", "").startswith("execute_context")
                    and (e.get("dur") or 0) > 1e5), key=lambda e: e["ts"])
    return [(f"step{i}_{s['name']}", s["ts"], s["ts"] + s["dur"]) for i, s in enumerate(steps)]

def union(iv):
    iv = sorted(iv); tot = 0; cs = ce = None
    for s, e in iv:
        if cs is None: cs, ce = s, e
        elif s <= ce: ce = max(ce, e)
        else: tot += ce - cs; cs, ce = s, e
    if cs is not None: tot += ce - cs
    return tot

def run(tp, syn, out):
    lines = []
    for name, start, end in windows(tp):
        per = collections.defaultdict(lambda: [0, 0, []])
        streams = collections.defaultdict(list)
        tot_cycles = 0
        with open(syn) as h:
            for line in h:
                if '"type:' not in line: continue
                try: e = json.loads(line.rstrip().rstrip(","))
                except ValueError: continue
                ts = e.get("ts")
                if ts is None or not (start <= ts <= end): continue
                m = CY.search((e.get("args") or {}).get("message", ""))
                if not m: continue
                c = sum(int(g) for g in m.groups())
                if c == 0: continue
                d = e.get("dur") or 0
                row = per[e["name"]]; row[0] += 1; row[1] += c; row[2].append((ts, ts + d))
                streams[e.get("pid")].append((ts, ts + d))
                tot_cycles += c
        span = end - start
        lines.append(f"### {name}  span={span/1000:.1f} ms  sum_cycles={tot_cycles:,}  "
                     f"dev_time={tot_cycles/1.4e9*1000:.1f} ms  wall_union={union([i for v in streams.values() for i in v])/1000:.1f} ms  streams={len(streams)}")
        for nm, (n, c, iv) in sorted(per.items(), key=lambda kv: -kv[1][1]):
            if c < tot_cycles * 0.005: continue
            lines.append(f"  {c:>13,} {100*c/tot_cycles:6.2f}%  n={n:<5} mean={c//n:>10,}  union={union(iv)/1000:8.1f}ms  {nm}")
        # stream lane breakdown for the two collectives
        for nm in ("custom_sync_all_machine", "custom_allreduce_ring_simple_bf16_rdma"):
            if nm in per:
                lane = collections.Counter()
                with open(syn) as h:
                    for line in h:
                        if nm not in line: continue
                        try: e = json.loads(line.rstrip().rstrip(","))
                        except ValueError: continue
                        if e.get("ts") is None or not (start <= e["ts"] <= end): continue
                        lane[e.get("pid")] += 1
                lines.append(f"  lanes[{nm}] = {lane.most_common()}")
        lines.append("")
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3])
