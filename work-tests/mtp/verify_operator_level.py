#!/usr/bin/env python3
"""Operator-level verification for GLM-5.3 MTP on DLC production geometry."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

CAPTURE_ROOT = Path(
    "/root/workspace/work-tests/glm53-flash/kpool-real-input-16k"
)
DEFAULT_BUNDLE = CAPTURE_ROOT / "capture_0001_real16k_t4096_seq16384"
RESULT_PATH = Path("/root/workspace/work-tests/mtp/results/operator_level.json")
DEVICE = torch.device("dlc:0")


def _load_bundle(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path / "tensors.pt", map_location="cpu", weights_only=True)


def _timed(fn, warmup: int = 2, iterations: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.dlc.synchronize()
    values = []
    for _ in range(iterations):
        started = time.perf_counter()
        fn()
        torch.dlc.synchronize()
        values.append((time.perf_counter() - started) * 1e3)
    return statistics.median(values)


def _select_topk_reference(
    query: torch.Tensor,
    head_weights: torch.Tensor,
    pool_cache: torch.Tensor,
    block_table: torch.Tensor,
    qsl: torch.Tensor,
    seq_lens: torch.Tensor,
    topk_tokens: int,
    pool_size: int,
) -> torch.Tensor:
    tokens, heads, head_dim = query.shape
    pools_per_block = pool_cache.shape[1]
    select_k = topk_tokens // pool_size
    out = torch.full(
        (tokens, 2176), -1, dtype=torch.int32, device=query.device
    )
    bounds = qsl.to(torch.int64)
    for row in range(tokens):
        token = torch.tensor(row, device=query.device)
        request = int(torch.searchsorted(bounds, token, right=True)) - 1
        start = int(bounds[request])
        query_len = int(bounds[request + 1]) - start
        before = int(seq_lens[request]) - query_len
        position = before + row - start
        if position < 0:
            continue
        seql = position + 1
        visible = seql // pool_size
        count = min(select_k, visible)
        if count:
            logical = torch.arange(visible, device=query.device)
            physical = block_table[request, logical // pools_per_block]
            pooled = pool_cache[physical, logical % pools_per_block]
            dots = torch.einsum(
                "hd,pd->hp", query[row].float(), pooled.float()
            ) * head_dim**-0.5
            dots = torch.relu(dots)
            scores = (
                dots
                * (head_weights[row].float() * heads**-0.5).unsqueeze(-1)
            ).sum(0)
            order = torch.argsort(-scores, stable=True)[:count]
            width = 0
            for pool in order.tolist():
                for offset in range(pool_size):
                    token = pool * pool_size + offset
                    out[row, width] = token if token < seql else -1
                    width += 1
        tail_len = seql % pool_size
        if tail_len:
            out[row, topk_tokens : topk_tokens + tail_len] = torch.arange(
                visible * pool_size, seql, dtype=torch.int32, device=query.device
            )
    return out


def verify_kpool(bundle_path: Path) -> dict[str, Any]:
    data = _load_bundle(bundle_path)
    query_full = data["query"].to(DEVICE)
    weights_full = data["head_weights"].to(DEVICE)
    pool = data["pool_cache"].to(DEVICE)
    table_full = data["block_table"].to(DEVICE)
    results: dict[str, Any] = {}

    for tokens in (4, 8):
        query = query_full[:tokens].contiguous()
        weights = weights_full[:tokens].contiguous()
        qsl = torch.tensor([0, tokens], dtype=torch.int32, device=DEVICE)
        seq_lens = torch.tensor([16384], dtype=torch.int32, device=DEVICE)
        out = torch.empty((tokens, 2176), dtype=torch.int32, device=DEVICE)
        reference = _select_topk_reference(
            query.cpu(),
            weights.cpu(),
            pool.cpu(),
            table_full.cpu(),
            qsl.cpu(),
            seq_lens.cpu(),
            2048,
            4,
        )

        def run(
            query=query,
            weights=weights,
            qsl=qsl,
            seq_lens=seq_lens,
            out=out,
        ) -> None:
            torch.dlc_vllm_kpool_select_topk(
                query,
                weights,
                pool,
                table_full,
                qsl,
                seq_lens,
                2048,
                4,
                None,
                None,
                out=out,
            )

        run()
        torch.dlc.synchronize()
        out_cpu = out.cpu()
        mismatch = int((out_cpu != reference).sum())
        pooled_out = out_cpu[:, :2048].reshape(tokens, 512, 4)[:, :, 0] // 4
        pooled_ref = reference[:, :2048].reshape(tokens, 512, 4)[:, :, 0] // 4
        sets_equal = all(
            set(pooled_out[row].tolist()) == set(pooled_ref[row].tolist())
            for row in range(tokens)
        )
        row_exact = True
        for row in range(tokens):
            row_out = torch.empty((1, 2176), dtype=torch.int32, device=DEVICE)
            torch.dlc_vllm_kpool_select_topk(
                query[row : row + 1],
                weights[row : row + 1],
                pool,
                table_full,
                torch.tensor([0, 1], dtype=torch.int32, device=DEVICE),
                torch.tensor(
                    [16384 - tokens + 1 + row], dtype=torch.int32, device=DEVICE
                ),
                2048,
                4,
                None,
                None,
                out=row_out,
            )
            torch.dlc.synchronize()
            row_exact = row_exact and torch.equal(
                out[row : row + 1].cpu(), row_out.cpu()
            )
        results[f"select_next_n_{tokens}"] = {
            "exact": mismatch == 0,
            "mismatch_elements": mismatch,
            "selected_pool_sets_equal": sets_equal,
            "rowwise_consistent": row_exact,
            "median_ms": _timed(run),
        }

    positions = torch.arange(16380, 16388, dtype=torch.int64, device=DEVICE)
    qsl = torch.tensor([0, 8], dtype=torch.int32, device=DEVICE)
    query_lens = torch.tensor([8], dtype=torch.int32, device=DEVICE)
    tail_pages = torch.tensor([7], dtype=torch.int32, device=DEVICE)
    live = torch.ones(8, dtype=torch.bool, device=DEVICE)
    req_indices = torch.zeros(1, dtype=torch.int64, device=DEVICE)

    def slot_mapping():
        return torch.dlc_kpool_slot_mapping(
            qsl,
            query_lens,
            positions,
            table_full,
            tail_pages,
            live,
            req_indices,
            4,
            64,
            8,
        )

    pool_slots, tail_slots = slot_mapping()
    expected_pool = [-1, -1, -1, -1, -1, -1, -1, -1]
    expected_pool[3] = int(table_full[0, 63]) * 64 + 63
    expected_pool[7] = int(table_full[0, 64]) * 64
    expected_tail = [7 * 4 + pos for pos in range(4)] * 2
    results["slot_mapping_spec_window"] = {
        "exact_pool": pool_slots.cpu().tolist() == expected_pool,
        "exact_tail": tail_slots.cpu().tolist() == expected_tail,
        "median_ms": _timed(slot_mapping),
    }

    compact = torch.arange(8 * 2176, dtype=torch.int32, device=DEVICE).view(8, 2176)
    compact_expected = compact.cpu().clone()
    slots = torch.tensor([1, 4, 7], dtype=torch.int64, device=DEVICE)
    compact_expected[:3] = compact_expected[slots.cpu()]

    def compact_topk() -> None:
        compact[:3] = compact[slots]

    compact_topk()
    torch.dlc.synchronize()
    results["compact_topk_production_width"] = {
        "exact": bool(torch.equal(compact.cpu(), compact_expected)),
        "median_ms": _timed(compact_topk),
    }
    return results


def verify_kpool_transaction() -> dict[str, Any]:
    layers = 11
    dim, pool_size, pools_per_block = 128, 4, 64
    bundles = sorted(CAPTURE_ROOT.glob("capture_*"))[:layers]
    pools = []
    tails = []
    for bundle in bundles:
        data = _load_bundle(bundle)
        pool = data["pool_cache"].to(DEVICE)
        blocks = int(pool.shape[0])
        pools.append(pool)
        tails.append(
            torch.zeros(
                (blocks, 2, pool_size, dim),
                dtype=torch.bfloat16,
                device=DEVICE,
            )
        )
    table = _load_bundle(bundles[0])["block_table"].to(DEVICE)
    pages = [int(table[0, 63]), int(table[0, 64])]
    tail_page = 7
    snapshot_pool = [
        (torch.empty_like(pool[pages[0]]), torch.empty_like(pool[pages[1]]))
        for pool in pools
    ]
    snapshot_tail = [
        torch.empty_like(tail[tail_page]) for tail in tails
    ]
    ape = torch.zeros((pool_size, dim), dtype=torch.float32, device=DEVICE)
    tail_table = torch.zeros((1, 1), dtype=torch.int32, device=DEVICE)
    positions = torch.arange(16380, 16388, device=DEVICE)
    tail_slots = (tail_page * pool_size + positions.remainder(pool_size)).to(torch.int32)
    pool_slots = torch.full((8,), -1, dtype=torch.int32, device=DEVICE)
    pool_slots[3] = pages[0] * pools_per_block + 63
    pool_slots[7] = pages[1] * pools_per_block
    keys = torch.randn((8, dim), dtype=torch.bfloat16, device=DEVICE) * 0.1
    gates = torch.randn((8, dim), dtype=torch.bfloat16, device=DEVICE) * 0.1
    candidate_qsl = torch.tensor([0, 8], dtype=torch.int32, device=DEVICE)
    candidate_seq = torch.tensor([16388], dtype=torch.int32, device=DEVICE)
    replay_qsl = torch.tensor([0, 5], dtype=torch.int32, device=DEVICE)
    replay_seq = torch.tensor([16385], dtype=torch.int32, device=DEVICE)

    reference_pool = pools[0].clone()
    reference_tail = tails[0].clone()
    torch.dlc_kpool_append_compress(
        keys[:5],
        gates[:5],
        ape,
        replay_qsl,
        replay_seq,
        tail_slots[:5],
        pool_slots[:5],
        tail_table,
        reference_tail,
        reference_pool,
        pool_size,
    )

    def append_all(qsl, seq, count: int) -> None:
        for pool, tail in zip(pools, tails):
            torch.dlc_kpool_append_compress(
                keys[:count],
                gates[:count],
                ape,
                qsl,
                seq,
                tail_slots[:count],
                pool_slots[:count],
                tail_table,
                tail,
                pool,
                pool_size,
            )

    def transaction_once() -> None:
        for index in range(layers):
            snapshot_pool[index][0].copy_(pools[index][pages[0]])
            snapshot_pool[index][1].copy_(pools[index][pages[1]])
            snapshot_tail[index].copy_(tails[index][tail_page])
        append_all(candidate_qsl, candidate_seq, 8)
        for index in range(layers):
            pools[index][pages[0]].copy_(snapshot_pool[index][0])
            pools[index][pages[1]].copy_(snapshot_pool[index][1])
            tails[index][tail_page].copy_(snapshot_tail[index])
        append_all(replay_qsl, replay_seq, 5)

    transaction_once()
    torch.dlc.synchronize()
    exact_pages = all(
        torch.equal(pools[0][page], reference_pool[page]) for page in pages
    )
    exact_tail = torch.equal(
        tails[0][tail_page], reference_tail[tail_page]
    )

    return {
        "layers": layers,
        "snapshot_pool_pages_per_layer": 2,
        "snapshot_tail_pages_per_layer": 1,
        "layer0_touched_pages_exact": exact_pages,
        "layer0_tail_page_exact": exact_tail,
        "median_ms": _timed(transaction_once, warmup=3, iterations=15),
    }


def verify_moe() -> dict[str, Any]:
    torch.manual_seed(29)
    tokens, hidden, intermediate, experts, topk = 4, 4096, 2048, 288, 8
    w13 = torch.empty(
        (experts, 2 * intermediate, hidden),
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    w2 = torch.empty(
        (experts, hidden, intermediate),
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    w13.normal_(0.0, 0.02)
    w2.normal_(0.0, 0.02)
    hidden_states = torch.randn(
        (tokens, hidden), dtype=torch.bfloat16, device=DEVICE
    )
    gating = torch.full(
        (tokens, experts), -10.0, dtype=torch.float32, device=DEVICE
    )
    gating[:, :topk] = torch.arange(topk, device=DEVICE, dtype=torch.float32)

    def run() -> torch.Tensor:
        return torch.dlc_vllm_fused_moe(
            hidden_states,
            w13,
            w2,
            gating,
            topk,
            experts,
            None,
            True,
            False,
            1,
            1,
            "sigmoid",
            2.5,
            None,
            False,
            None,
            None,
            "silu",
            None,
        )

    got = run()
    torch.dlc.synchronize()
    selected_w13 = w13[:topk].cpu()
    selected_w2 = w2[:topk].cpu()
    hidden_cpu = hidden_states.cpu()
    scores = torch.sigmoid(gating.cpu().float())
    top_values = scores[:, :topk]
    top_values = top_values / top_values.sum(-1, keepdim=True) * 2.5
    expected = torch.zeros((tokens, hidden), dtype=torch.float32)
    for token in range(tokens):
        x = hidden_cpu[token].float()
        for expert in range(topk):
            gate, up = selected_w13[expert].float().chunk(2, dim=0)
            expert_out = F.silu(F.linear(x, gate)) * F.linear(x, up)
            expected[token] += top_values[token, expert] * F.linear(
                expert_out, selected_w2[expert].float()
            )
    expected = expected.to(torch.bfloat16)
    error = (got.cpu().float() - expected.float()).abs().max().item()
    relative = error / expected.float().abs().max().item()
    return {
        "shape": [tokens, hidden],
        "experts": experts,
        "topk": topk,
        "max_abs_error": error,
        "relative_error": relative,
        "median_ms": _timed(run, warmup=1, iterations=3),
        "allocated_gib": torch.dlc.memory_allocated(DEVICE) / 2**30,
    }


def verify_fusion() -> dict[str, Any]:
    from vllm_cl.patch.models.patch_glm5_mtp import fused_eh_norm_dlc

    torch.manual_seed(37)
    tokens, hidden = 4, 4096
    positions = torch.arange(tokens, dtype=torch.int64, device=DEVICE)
    positions[0] = 0
    embeds = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=DEVICE)
    previous = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=DEVICE)
    enorm = torch.randn((hidden,), dtype=torch.bfloat16, device=DEVICE)
    hnorm = torch.randn((hidden,), dtype=torch.bfloat16, device=DEVICE)

    def run() -> torch.Tensor:
        return fused_eh_norm_dlc(
            positions, embeds, previous, enorm, hnorm, 1e-6
        )

    got = run()
    torch.dlc.synchronize()
    return {
        "shape": list(got.shape),
        "position_zero_masked": bool(
            torch.equal(
                got[0, :hidden].cpu(),
                torch.zeros(hidden, dtype=got.dtype),
            )
        ),
        "median_ms": _timed(run, warmup=2, iterations=10),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sections",
        default="kpool,transaction,moe,fusion",
        help="Comma-separated: kpool,transaction,moe,fusion",
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    torch.dlc.set_device(DEVICE)
    results: dict[str, Any] = {}
    sections = {item.strip() for item in args.sections.split(",") if item.strip()}
    if "kpool" in sections:
        results["kpool"] = verify_kpool(args.bundle)
    if "transaction" in sections:
        results["kpool_transaction"] = verify_kpool_transaction()
    if "moe" in sections:
        results["moe"] = verify_moe()
    if "fusion" in sections:
        results["fusion"] = verify_fusion()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
