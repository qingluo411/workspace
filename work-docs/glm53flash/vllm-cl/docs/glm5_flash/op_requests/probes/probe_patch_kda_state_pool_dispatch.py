#!/usr/bin/env python3
"""Verify the `vllm_cl.patch.patch_kda` wiring for #05 (kernel #8021 / pytorch #2154).

Runs the *real* dispatcher (`gather_initial_states_dlc` / `scatter_states_dlc`) that the
model calls, on the real GLM-5.3-Flash pool geometry, and checks:
  1. default mode (`DLC_KDA_STATE_POOL_OP=kernel`) routes to
     `torch.dlc_kda_state_gather/_scatter` and is bit-exact vs the plugin page-table path;
  2. `fast` / `native` modes still work and agree;
  3. preconditions keep the fallback honest: a non-contiguous multi-column block table,
     a bf16 `src`, an int64 index vector all fall back instead of raising;
  4. `validate_data=True` still raises for out-of-range / duplicate pages.

Must be run from a cwd other than `/root/workspace`, otherwise the namespace dir
`/root/workspace/vllm` shadows the editable `vllm` package.

Usage: python -u probe_patch_kda_state_pool_dispatch.py [dlc:4]
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

for candidate in (Path(__file__).resolve().parents[4], Path("/root/workspace/vllm-cl")):
    if (candidate / "vllm_cl").is_dir():
        sys.path.insert(0, str(candidate))
        break

import vllm  # noqa: F401,E402  必须先把真正的 vllm 包导进来
from vllm_cl.patch import patch_kda  # noqa: E402
from vllm_cl.patch.kda_state_pool_dlc import (  # noqa: E402
    gather_initial_states_dlc as plugin_gather,
    scatter_states_dlc as plugin_scatter,
)

PAD = -1
H, DV, DK = 8, 128, 128
PAGES = 1168
PAGE_LEN = 135680
REC_OFF = PAGE_LEN - H * DV * DK


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


def main():
    dev_name = sys.argv[1] if len(sys.argv) > 1 else "dlc:4"
    torch.dlc.set_device(int(dev_name.split(":")[-1]))
    print(f"device={dev_name} mode={patch_kda._STATE_POOL_OP_MODE}", flush=True)

    ck = Checker()
    torch.manual_seed(0)
    backing = (torch.randn(PAGES * PAGE_LEN, dtype=torch.float32) * 3.0).reshape(PAGES, PAGE_LEN)
    dev = backing.to(dev_name)
    state = dev.as_strided((PAGES, H, DV, DK), (PAGE_LEN, DV * DK, DK, 1), REC_OFF)
    state_cpu = backing.as_strided((PAGES, H, DV, DK), (PAGE_LEN, DV * DK, DK, 1), REC_OFF)
    pristine = dev.clone()

    def golden_gather(indices, mask):
        out = torch.zeros((indices.numel(), H, DV, DK), dtype=torch.float32)
        for s in range(indices.numel()):
            p = int(indices[s])
            if bool(mask[s]) and 0 <= p < PAGES:
                out[s].copy_(state_cpu[p])
        return out

    # 1. default mode must go through the fused op
    calls = {"gather": 0, "scatter": 0}
    real_gather, real_scatter = torch.dlc_kda_state_gather, torch.dlc_kda_state_scatter

    def counting_gather(*a, **kw):
        calls["gather"] += 1
        return real_gather(*a, **kw)

    def counting_scatter(*a, **kw):
        calls["scatter"] += 1
        return real_scatter(*a, **kw)

    torch.dlc_kda_state_gather, torch.dlc_kda_state_scatter = counting_gather, counting_scatter
    ck.check("default mode is kernel", patch_kda._STATE_POOL_OP_MODE == "kernel", patch_kda._STATE_POOL_OP_MODE)

    idx = torch.tensor([0, 97, PAD, 1167], dtype=torch.int32, device=dev_name)
    mask = torch.tensor([True, False, False, True], dtype=torch.bool, device=dev_name)
    want = golden_gather(torch.tensor([0, 97, PAD, 1167]), torch.tensor([True, False, False, True]))
    ck.equal("kernel mode gather == golden", patch_kda.gather_initial_states_dlc(state, idx, mask).cpu(), want)
    ck.check("kernel mode gather hit the op", calls["gather"] == 1, str(calls))

    src = (torch.randn((4, H, DV, DK)) * 2.0).to(dev_name)
    patch_kda.scatter_states_dlc(state, src, idx)
    ck.check("kernel mode scatter hit the op", calls["scatter"] == 1, str(calls))
    rows = dev.index_select(0, torch.tensor([0, 97, 1167], device=dev_name).long())
    got = rows.narrow(1, REC_OFF, H * DV * DK).reshape(3, H, DV, DK).cpu()
    ck.equal("kernel mode scatter wrote rows", got, src[[0, 1, 3]].cpu())
    untouched = dev.index_select(0, torch.tensor([5, 900], device=dev_name).long())
    ck.equal("kernel mode scatter left others alone",
             untouched.narrow(1, REC_OFF, H * DV * DK).reshape(2, H, DV, DK).cpu(),
             torch.stack([state_cpu[5], state_cpu[900]]))

    # 2. fallback modes agree; the op must not be called
    # (`native` rejects any negative index, so the fallback comparison uses PAD-free rows)
    idx_ok = torch.tensor([0, 97, 1167, 5], dtype=torch.int32, device=dev_name)
    mask_ok = torch.tensor([True, False, True, True], dtype=torch.bool, device=dev_name)
    want_ok = golden_gather(torch.tensor([0, 97, 1167, 5]), torch.tensor([True, False, True, True]))
    dev.copy_(pristine)
    before = dict(calls)
    for mode in ("fast", "native"):
        patch_kda._STATE_POOL_OP_MODE = mode
        ck.equal(f"{mode} mode gather == golden",
                 patch_kda.gather_initial_states_dlc(state, idx_ok, mask_ok).cpu(), want_ok)
        ck.equal(f"{mode} mode gather == plugin",
                 patch_kda.gather_initial_states_dlc(state, idx_ok, mask_ok).cpu(),
                 plugin_gather(state, idx_ok, mask_ok).cpu())
    ck.check("fallback modes did not call the op", calls == before, f"{before} -> {calls}")

    # 3. precondition failures must fall back, not raise
    patch_kda._STATE_POOL_OP_MODE = "kernel"
    bt = torch.zeros(128, 2, dtype=torch.int32, device=dev_name)
    bt[:4, 1] = idx_ok.to(dev_name)
    col = bt[:4, 1]
    ck.check("multi-column column view is non-contiguous", not col.is_contiguous())
    ck.equal("non-contiguous indices fall back to page table",
             patch_kda.gather_initial_states_dlc(state, col, mask_ok).cpu(), want_ok)
    ck.check("non-contiguous indices did not call the op", calls == before, str(calls))
    idx64 = idx.to(torch.int64)
    ck.equal("int64 indices == golden", patch_kda.gather_initial_states_dlc(state, idx64, mask).cpu(), want)
    ck.check("int64 indices did call the op", calls["gather"] == before["gather"] + 1, str(calls))
    ck.equal("int32 mask == golden", patch_kda.gather_initial_states_dlc(state, idx, mask.to(torch.int32)).cpu(), want)

    dev.copy_(pristine)
    src_bf16 = src.to(torch.bfloat16)
    ck.check("bf16 src is rejected by the op guard", src_bf16.dtype != state.dtype)
    patch_kda.scatter_states_dlc(state, src_bf16, idx)
    rows = dev.index_select(0, torch.tensor([0, 97, 1167], device=dev_name).long())
    ck.equal("bf16 src scatter falls back and casts",
             rows.narrow(1, REC_OFF, H * DV * DK).reshape(3, H, DV, DK).cpu(),
             src_bf16[[0, 1, 3]].to(torch.float32).cpu())

    # 4. validate_data still raises
    for pages, name in (([0, PAGES], "out of range"), ([3, 3], "duplicate")):
        try:
            patch_kda.scatter_states_dlc(
                state, src, torch.tensor(pages, dtype=torch.int32, device=dev_name), True)
            ck.check(f"validate_data raises for {name}", False, "no exception")
        except Exception as error:  # noqa: BLE001
            ck.check(f"validate_data raises for {name}", True)

    return ck.summary()


if __name__ == "__main__":
    sys.exit(main())
