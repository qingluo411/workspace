#!/usr/bin/env python3
"""Verify `torch.dlc_kda_state_gather/_scatter` (kernel #8021 / pytorch #2154).

Real GLM-5.3-Flash geometry: P=1168 pages, page = 135680 fp32 (conv 4608 +
recurrent 131072), i.e. the `[P, 8, 128, 128]` view has storage_offset 4608 and
stride(0) 135680 (dim0 non-contiguous, 4608 floats of conv head per page).

Checks:
  1. gather/scatter vs the CPU golden of the request doc (`05_...md` semantics);
  2. gather/scatter vs the plugin page-table implementation (current production
     path, `vllm_cl/patch/kda_state_pool_dlc.py`);
  3. dtype/layout variants: int32+int64 indices, bool+int32 mask, `out=` buffer,
     and the `block_table[:, 0]` column view the model actually passes;
  4. scatter only writes the selected pages (sampled untouched pages compared
     through the contiguous 2-D table, never by fancy-indexing the 4-D view);
  5. graph capture + replay after the indices buffer is rewritten;
  6. `--perf`: per-call cost vs the plugin implementation, at S=1/8/64/512;
  7. `--device-errors`: validate_data assertions, in subprocesses.

Usage: python -u probe_kda_state_gather_scatter_op.py [dlc:0] [--perf] [--device-errors]
Exit code 0 = all enabled checks pass.
"""
from __future__ import annotations

import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

for candidate in (Path(__file__).resolve().parents[4], Path("/root/workspace/vllm-cl")):
    if (candidate / "vllm_cl").is_dir():
        sys.path.insert(0, str(candidate))
        break
from vllm_cl.patch.kda_state_pool_dlc import (  # noqa: E402
    gather_initial_states_dlc as plugin_gather,
    page_table,
    scatter_states_dlc as plugin_scatter,
)

PAD = -1
H, DV, DK = 8, 128, 128
PAGES = 1168
PAGE_LEN = 135680
REC_OFF = PAGE_LEN - H * DV * DK  # 4608


class Checker:
    def __init__(self):
        self.failed = []

    def check(self, name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f"  {detail}"), flush=True)
        if not ok:
            self.failed.append(name)

    def equal(self, name, got, want):
        ok = tuple(got.shape) == tuple(want.shape) and bool(torch.equal(got, want))
        self.check(name, ok, "shape/bit mismatch" if not ok else "")

    def summary(self):
        print()
        if self.failed:
            print(f"RESULT: FAIL ({len(self.failed)}): " + ", ".join(self.failed[:6]))
            return 1
        print("RESULT: PASS")
        return 0


def build(P=PAGES, device="dlc:0", seed=0):
    torch.manual_seed(seed)
    backing = (torch.randn(P * PAGE_LEN, dtype=torch.float32) * 3.0)
    backing = backing.reshape(P, PAGE_LEN)
    dev = backing.to(device)
    view = dev.as_strided((P, H, DV, DK), (PAGE_LEN, DV * DK, DK, 1), REC_OFF)
    cpu = backing.as_strided((P, H, DV, DK), (PAGE_LEN, DV * DK, DK, 1), REC_OFF)
    return dev, view, cpu, backing


def ref_gather(state_cpu, indices, mask):
    out = torch.zeros((indices.numel(), H, DV, DK), dtype=torch.float32)
    for s in range(indices.numel()):
        p = int(indices[s])
        if bool(mask[s]) and 0 <= p < state_cpu.shape[0]:
            out[s].copy_(state_cpu[p])
    return out


def read_rec(table, pages):
    """Read the recurrent window of `pages` through the contiguous 2-D table."""
    rows = table.index_select(0, torch.as_tensor(pages, dtype=torch.int64, device=table.device))
    return rows.narrow(1, REC_OFF, H * DV * DK).reshape(len(pages), H, DV, DK).cpu()


def main():
    argv = sys.argv[1:]
    flags = {a for a in argv if a.startswith("--")}
    args = [a for a in argv if not a.startswith("--")]
    dev_name = args[0] if args else "dlc:0"
    torch.dlc.set_device(int(dev_name.split(":")[-1]))
    print(f"device={dev_name} P={PAGES} page_len={PAGE_LEN} rec_off={REC_OFF} pool={PAGES*PAGE_LEN*4/2**20:.0f} MiB",
          flush=True)

    ck = Checker()
    device, state, state_cpu, cpu_backing = build(device=dev_name)
    table = device  # [P, PAGE_LEN] contiguous
    pristine = device.clone()  # restore between sections: every check is against the pristine pool
    ck.check("geometry is the model form",
             state.storage_offset() == REC_OFF and state.stride(0) == PAGE_LEN)

    def restore():
        device.copy_(pristine)

    print("\n== 1/2/3 correctness vs golden and vs plugin ==", flush=True)
    restore()
    for S in (1, 2, 8, 64):
        rng = torch.Generator().manual_seed(S)
        base = torch.arange(S, dtype=torch.int64).mul(97).remainder(PAGES)
        idx_cpu = base.clone()
        mask_cpu = torch.ones(S, dtype=torch.bool)
        if S > 1:
            idx_cpu[1::3] = PAD
            mask_cpu[1::3] = False
            idx_cpu[2::5] = torch.randint(0, PAGES, (len(idx_cpu[2::5]),), generator=rng)
            mask_cpu[2::5] = False
            idx_cpu[-1] = PAGES - 1
            mask_cpu[-1] = True
        restore()
        for i64 in (False, True):
            for bool_mask in (False, True):
                restore()
                idx = idx_cpu.to(torch.int64 if i64 else torch.int32).to(dev_name)
                mask = mask_cpu.to(torch.bool if bool_mask else torch.int32).to(dev_name)
                tag = f"S={S} idx={'i64' if i64 else 'i32'} mask={'bool' if bool_mask else 'i32'}"
                want = ref_gather(state_cpu, idx_cpu, mask_cpu)
                got = torch.dlc_kda_state_gather(state, idx, mask)
                ck.equal(f"gather/golden {tag}", got.cpu(), want)
                out = torch.empty_like(got)
                ck.check(f"out= returns the buffer {tag}", torch.dlc_kda_state_gather(state, idx, mask, out=out) is out)
                ck.equal(f"gather/out= {tag}", out.cpu(), want)
                ck.equal(f"gather/plugin {tag}", plugin_gather(state, idx, mask).cpu(), want)

                src = (torch.randn((S, H, DV, DK), generator=rng) * 2.0).to(dev_name)
                torch.dlc_kda_state_scatter(state, src, idx)
                wrote = torch.as_tensor([int(i) for i in idx_cpu if int(i) >= 0 and int(mask_cpu[list(idx_cpu).index(i)])] if False else
                                        sorted({int(i) for i in idx_cpu if i >= 0}), dtype=torch.int64)
                got_rows = read_rec(table, wrote.tolist()).to(dev_name)
                src_map = {int(idx_cpu[s]): src[s] for s in range(S) if int(idx_cpu[s]) >= 0}
                want_rows = torch.stack([src_map[int(p)] for p in wrote]).cpu()
                ck.equal(f"scatter wrote pages {tag}", got_rows.cpu(), want_rows)
                untouched = [p for p in (7, 101, 500, 900, 1166) if p not in src_map]
                if untouched:
                    ck.equal(f"scatter untouched {tag}", read_rec(table, untouched),
                             torch.stack([state_cpu[p] for p in untouched]))

    print("\n== 3b block-table column view (what the model passes) ==", flush=True)
    restore()
    B = 8
    ids = torch.tensor([900, 3, 17, 1167, 0, 55, 2, 8], dtype=torch.int32)
    bt = torch.zeros(128, 1, dtype=torch.int32, device=dev_name)
    bt[:B, 0] = ids.to(dev_name)
    col = bt[:B, 0]
    mask = torch.ones(B, dtype=torch.bool, device=dev_name)
    want = ref_gather(state_cpu, ids, torch.ones(B, dtype=torch.bool))
    ck.equal("gather block_table[:, 0] (i32 row-padded view)",
             torch.dlc_kda_state_gather(state, col, mask).cpu(), want)
    bt64 = torch.zeros(128, 2, dtype=torch.int64, device=dev_name)
    bt64[:B, 1] = ids.to(torch.int64).to(dev_name)
    try:
        got64 = torch.dlc_kda_state_gather(state, bt64[:B, 1], mask)
        ck.equal("gather 2-D table column (i64, non-contiguous)",
                 got64.cpu(), ref_gather(state_cpu, ids.to(torch.int64), torch.ones(B, dtype=torch.bool)))
    except RuntimeError as error:
        ck.check("gather non-contiguous column rejected (expected: model config is 1 column)",
                 "must be a same-device contiguous" in str(error), str(error)[:120])
    i32mask = torch.ones(B, dtype=torch.int32, device=dev_name)
    ck.equal("gather int32 mask",
             torch.dlc_kda_state_gather(state, col, i32mask).cpu(), want)

    print("\n== 4 semantics: PAD(-1) / null page / padded tail ==", flush=True)
    restore()
    pad_idx = torch.tensor([PAD, PAD], dtype=torch.int32, device=dev_name)
    ck.equal("gather PAD(mask=True) -> zeros", torch.dlc_kda_state_gather(state, pad_idx, torch.ones(2, dtype=torch.bool, device=dev_name)).cpu(), torch.zeros((2, H, DV, DK), dtype=torch.float32))
    ck.equal("gather PAD(mask=False) -> zeros", torch.dlc_kda_state_gather(state, pad_idx, torch.zeros(2, dtype=torch.bool, device=dev_name)).cpu(), torch.zeros((2, H, DV, DK), dtype=torch.float32))
    torch.dlc_kda_state_scatter(state, torch.full((2, H, DV, DK), 9.0, device=dev_name), pad_idx)
    torch.dlc.synchronize()
    ck.check("scatter PAD-only leaves the pool bit-identical", bool(torch.equal(device, pristine)))
    # the model's real call: block_table[:, 0] with a NULL_BLOCK_ID(0) padded tail + has=False tail
    ids = torch.tensor([7, 900, 0, 0], dtype=torch.int32, device=dev_name)
    has = torch.tensor([True, True, False, False], dtype=torch.bool, device=dev_name)
    ck.equal("gather padded tail (NULL=0, has=False) == plugin", torch.dlc_kda_state_gather(state, ids, has).cpu(), plugin_gather(state, ids, has).cpu())

    print("\n== 5 graph capture / replay ==", flush=True)
    idx = torch.tensor([5, 6], dtype=torch.int32, device=dev_name)
    mask2 = torch.ones(2, dtype=torch.bool, device=dev_name)
    buf = torch.empty(2, H, DV, DK, dtype=torch.float32, device=dev_name)
    src = torch.full((2, H, DV, DK), 7.0, device=dev_name)
    if hasattr(torch.dlc, "DLCGraph"):
        graph = torch.dlc.DLCGraph()
        with torch.dlc.graph(graph):
            alloc = torch.dlc_kda_state_gather(state, idx, mask2)
            torch.dlc_kda_state_gather(state, idx, mask2, out=buf)
            torch.dlc_kda_state_scatter(state, src, idx)
        ptr = alloc.data_ptr()
        for ids2 in ([5, 6], [11, PAD], [1000, 3]):
            ci = torch.tensor(ids2, dtype=torch.int32)
            idx.copy_(ci.to(dev_name))
            graph.replay()
            one = torch.ones(2, dtype=torch.bool)
            ck.check(f"replay keeps allocation {ids2}", alloc.data_ptr() == ptr)
            ck.equal(f"replay gather {ids2}", alloc.cpu(), ref_gather(state_cpu, ci, one))
            ck.equal(f"replay gather(out=) {ids2}", buf.cpu(), ref_gather(state_cpu, ci, one))
        seven = torch.full((H, DV, DK), 7.0)
        for p in (5, 6, 1000):
            ck.equal(f"replay scatter page {p}", read_rec(table, [p])[0], seven)
    else:
        ck.check("DLCGraph available", False, "torch.dlc.DLCGraph missing")

    if "--perf" in flags:
        print("\n== 6 perf (real geometry; 34x = one decode step over 34 KDA layers) ==", flush=True)
        plug_table, _, _ = page_table(state)
        for S in (1, 8, 64, 128, 512):
            idx = (torch.arange(S, dtype=torch.int32, device=dev_name).remainder(PAGES - 1) + 1)
            mask = torch.ones(S, dtype=torch.bool, device=dev_name)
            out = torch.empty(S, H, DV, DK, dtype=torch.float32, device=dev_name)
            src = torch.full_like(out, 1.5)

            def bench(fn, iters=34, scale=1.0):
                for _ in range(3):
                    fn()
                torch.dlc.synchronize()
                samples = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    for _ in range(iters):
                        fn()
                    torch.dlc.synchronize()
                    samples.append((time.perf_counter() - t0) * 1000 / iters * scale)
                return statistics.median(samples)

            g_out = bench(lambda: torch.dlc_kda_state_gather(state, idx, mask, out=out))
            g_new = bench(lambda: torch.dlc_kda_state_gather(state, idx, mask))
            sc = bench(lambda: torch.dlc_kda_state_scatter(state, src, idx))
            pair = bench(lambda: (torch.dlc_kda_state_gather(state, idx, mask, out=out),
                                  torch.dlc_kda_state_scatter(state, src, idx)), iters=34, scale=34.0)
            p_g = bench(lambda: plugin_gather(state, idx, mask, table=plug_table))
            p_s = bench(lambda: plugin_scatter(state, src, idx, table=plug_table))
            print(f"  S={S:<4} op  gather(out=) {g_out:6.3f}  gather() {g_new:6.3f}  scatter {sc:6.3f}  "
                  f"34 pairs {pair:7.3f} ms | plugin gather {p_g:6.3f} scatter {p_s:6.3f} ms", flush=True)
            if S == 1:
                ck.check("S=1 gather <= 0.5 ms", g_out <= 0.5, f"{g_out:.3f}")
                ck.check("S=1 scatter <= 1.0 ms", sc <= 1.0, f"{sc:.3f}")
                ck.check("34 pairs <= 50 ms", pair <= 50.0, f"{pair:.3f}")

    if "--device-errors" in flags:
        print("\n== 7 validate_data device assertions (subprocess) ==", flush=True)
        for mode in ("range", "duplicate", "int64"):
            r = subprocess.run([sys.executable, __file__, dev_name, "--device-error", mode],
                               capture_output=True, text=True, timeout=180)
            ck.check(f"validate_data raises for {mode}", r.returncode != 0, (r.stdout + r.stderr)[-160:])

    return ck.summary()


def device_error(mode, dev_name="dlc:0"):
    _, st, _ = build(P=4, device=dev_name)
    ids = {"range": [0, 4], "duplicate": [0, 0], "int64": [0, 1 << 32]}[mode]
    torch.dlc_kda_state_scatter(st, torch.ones(2, H, DV, DK, device=st.device),
                                torch.tensor(ids, dtype=torch.int64, device=st.device),
                                validate_data=True)
    torch.dlc.synchronize()


if __name__ == "__main__":
    if "--device-error" in sys.argv:
        _rest = sys.argv[1:]
        _mode = _rest[_rest.index("--device-error") + 1]
        _dev = next((a for a in _rest if a.startswith("dlc:")), "dlc:0")
        device_error(_mode, _dev)
    else:
        sys.exit(main())
