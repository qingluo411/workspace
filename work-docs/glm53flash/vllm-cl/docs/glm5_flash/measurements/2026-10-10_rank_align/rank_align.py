#!/usr/bin/env python3
"""8-rank host step-alignment + per-step barrier-wait distribution.

用法：
    python rank_align.py '<glob of the 8 rank traces>' [output.txt]

数据源：同一轮采集的 8 个 rank torch trace（同一台机器、同一个 µs 时钟）。
窗口/口径见 docs/glm5_flash/集合通信同步归因_2026-10-10.md「追加 2」。

输出：
  表 1  各 rank 的 prefill 起点偏移 / decode 起点偏移 / decode 步长
  表 2  各 rank 在最后一个 decode 步里 87 次 custom_sync_all_machine 的时长分布
"""
import glob
import gzip
import json
import statistics as st
import sys

pat = sys.argv[1] if len(sys.argv) > 1 else \
    "/tmp/syn_log/prev_1152/dp0_pp0_tp*_dcp0_ep*_rank*.179160429*.pt.trace.json.gz"
out = open(sys.argv[2], "w") if len(sys.argv) > 2 else sys.stdout


def step_dur(trace):
    ev = json.load(gzip.open(trace))["traceEvents"]
    X = [e for e in ev if e.get("ph") == "X"]
    steps = sorted(
        (e for e in X if e.get("cat") == "user_annotation" and e["name"].startswith("execute_context")),
        key=lambda e: e["ts"],
    )
    pre = [s for s in steps if "generation_0" in s["name"] and (s.get("dur") or 0) > 0]
    dec = [s for s in steps if "generation_1" in s["name"] and (s.get("dur") or 0) > 0]
    return X, pre, dec


rows = []
for f in sorted(glob.glob(pat), key=lambda p: int(p.split("rank")[-1].split(".")[0])):
    rank = int(f.split("rank")[-1].split(".")[0])
    X, pre, dec = step_dur(f)
    rows.append(dict(rank=rank, X=X, pre=pre, dec=dec))
rows.sort(key=lambda r: r["rank"])
base = rows[0]

print("== 表1：8 rank 步边界对齐 ==", file=out)
print(f"{'rank':>4} | {'prefill_start(rel ms)':>21} | {'decode_start(rel ms)':>30} | decode_dur(ms)", file=out)
for r in rows:
    ps = (r["pre"][0]["ts"] - base["pre"][0]["ts"]) / 1000 if r["pre"] and base["pre"] else float("nan")
    base_dec0 = r["dec"][0]["ts"] if r["dec"] else 0
    ds = " ".join(f"{(s['ts'] - base_dec0) / 1000:+7.1f}" for s in r["dec"])
    dm = " ".join(f"{s['dur'] / 1000:6.1f}" for s in r["dec"])
    print(f"{r['rank']:>4} | {ps:>21.1f} | {ds:>30} | {dm}", file=out)

print("\n== 表2：最后一个 decode 步里 87 次 custom_sync_all_machine 的时长分布（host 侧）==", file=out)
print(f"{'rank':>4} {'N':>4} {'sum_ms':>8} {'min':>7} {'median':>7} {'p90':>7} {'max':>7} {'>512us_sum':>11}",
      file=out)
for r in rows:
    if not r["dec"]:
        print(f"{r['rank']:>4}  (no decode step annotation)", file=out)
        continue
    d = r["dec"][-1]
    a, b = d["ts"], d["ts"] + d["dur"]
    sy = sorted((e.get("dur") or 0) / 1000
                for e in r["X"] if e["name"] == "custom_sync_all_machine" and a <= e["ts"] <= b)
    if not sy:
        print(f"{r['rank']:>4}  (no sync event in window; trace may be truncated)", file=out)
        continue
    p90 = sy[min(int(0.9 * len(sy)), len(sy) - 1)]
    print(f"{r['rank']:>4} {len(sy):>4} {sum(sy):>8.1f} {sy[0]:>7.3f} {st.median(sy):>7.3f} "
          f"{p90:>7.3f} {sy[-1]:>7.3f} {sum(x for x in sy if x > 0.512):>11.1f}", file=out)
if out is not sys.stdout:
    out.close()
