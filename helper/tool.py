from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


FREQUENCY_MHZ = 1400
ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
ADDRESS = re.compile(r"addr=0x(\w+) ")
LAUNCH = re.compile(rb" launch\s+([^\s]+)\s+on dlc:(\d+)")
CYCLE = re.compile(
    rb"cycles:\s*(\d+)\s+ops:\s*(\d+)\s+bytes:\s*(\d+)\s+crt:\s*(\d+)"
)
CYCLE_ID = re.compile(
    rb"batch_id:\s*(\d+)\s+kernel_index:\s*(\d+)\s+"
    rb"kernel_name:\s*([^\s]+)\s+cycles:\s*(\d+)\s+ops:\s*(\d+)\s+"
    rb"bytes:\s*(\d+)\s+crt:\s*(\d+)"
)
XYS = re.compile(
    rb"(?:(?:batch_id:\s*(\d+)\s+kernel_index:\s*(\d+)\s+"
    rb"kernel_name:\s*([^\s]+)\s+)|)(?:([^\s]+)\s+)?"
    rb"xys0_cycle:\s*(\d+)\s+xys1_cycle:\s*(\d+)"
)
CYCLE_MARKER = re.compile(rb"^\s*(?:\xe2\x94\x80)+\s+cycles\s*$")


def remove_ansi(text: str) -> str:
    return ADDRESS.sub("xxx", ANSI_ESCAPE.sub("", text))


@dataclass
class Metric:
    name: str
    body: str
    cycles: int
    ops: int
    bytes: int
    crt: int
    batch_id: int | None = None
    kernel_index: int | None = None
    xys0: int | None = None
    xys1: int | None = None
    n: int = 1

    @property
    def count(self) -> int:
        return self.n

    def __iadd__(self, other: "Metric") -> "Metric":
        self.cycles += other.cycles
        self.ops += other.ops
        self.bytes += other.bytes
        self.crt += other.crt
        self.n += other.n
        return self

    def format(self, total: "Metric") -> str:
        cycles = max(self.cycles, 1)
        ops = max(self.ops, 1)
        byte_count = max(self.bytes, 1)
        crt = max(self.crt, 1)
        average = cycles / FREQUENCY_MHZ / self.count if self.count else 0
        crt_average = crt / FREQUENCY_MHZ / self.count if self.count else 0
        return (
            f"n: {self.count:<8}cycles({cycles / max(total.cycles, cycles) * 100:>6.2f}%): "
            f"{cycles:<12}  ops({ops / max(total.ops, ops) * 100:>6.2f}%):{ops:<16} "
            f"GFLOPS:{ops * FREQUENCY_MHZ / 1000 / cycles:>9.2f} "
            f"BAND:{byte_count * FREQUENCY_MHZ / 1000 / cycles:>8.2f} GB/s "
            f"t_avg:{average:>9.2f} us  crt_avg:{crt_average:>5.2f} us "
            f"t:{cycles / FREQUENCY_MHZ / 1000:>10.2f} ms "
            f"crt:{crt / FREQUENCY_MHZ / 1000:>8.2f} ms  {self.name}"
        )


@dataclass
class Diagnostics:
    total_cycles: int = 0
    parsed_events: int = 0
    id_events: int = 0
    fallback_events: int = 0
    name_mismatches: int = 0
    cycle_mismatches: int = 0
    unmatched_xys: int = 0
    unmatched_cycles: int = 0
    unprofiled_launches: int = 0


@dataclass
class ForwardRound:
    number: int
    name: str
    metrics: list[Metric]


def parse_cycle(line: bytes, source_name: str, body: str) -> Metric | None:
    identified = CYCLE_ID.search(line)
    if identified:
        batch_id, kernel_index, kernel_name, cycles, ops, byte_count, crt = identified.groups()
        return Metric(
            kernel_name.decode(), body, int(cycles), int(ops), int(byte_count), int(crt),
            int(batch_id), int(kernel_index),
        )
    match = CYCLE.search(line)
    if not match:
        return None
    cycles, ops, byte_count, crt = map(int, match.groups())
    return Metric(source_name, body, cycles, ops, byte_count, crt)


def parse_xys(line: bytes) -> tuple[int | None, int | None, str, str, int, int] | None:
    match = XYS.search(line)
    if not match:
        return None
    batch_id, kernel_index, identified_name, displayed_name, xys0, xys1 = match.groups()
    kernel_name = identified_name or displayed_name
    if kernel_name is None:
        return None
    return (
        int(batch_id) if batch_id is not None else None,
        int(kernel_index) if kernel_index is not None else None,
        kernel_name.decode(),
        displayed_name.decode() if displayed_name is not None else kernel_name.decode(),
        int(xys0),
        int(xys1),
    )


def metric_key(metric: Metric) -> tuple[int, int] | None:
    if metric.batch_id is None or metric.kernel_index is None:
        return None
    return metric.batch_id, metric.kernel_index


def resolve_metric(
    metric: Metric,
    xys_name: str,
    xys0: int,
    xys1: int,
    pending_by_id: dict[tuple[int, int], Metric],
    pending_by_name_cycle: dict[tuple[str, int], deque[Metric]],
    pending_by_cycle: dict[int, deque[Metric]],
    diagnostics: Diagnostics,
) -> Metric | None:
    resolved = None
    if metric_key(metric) is not None:
        resolved = pending_by_id.pop(metric_key(metric), None)
        if resolved is not None:
            diagnostics.id_events += 1
    if resolved is None:
        candidates = pending_by_name_cycle.get((xys_name, xys0))
        if candidates:
            resolved = candidates.popleft()
            diagnostics.fallback_events += 1
    if resolved is None:
        candidates = pending_by_cycle.get(xys0)
        if candidates:
            resolved = candidates.popleft()
            diagnostics.fallback_events += 1
    if resolved is None:
        diagnostics.unmatched_xys += 1
        return None
    key = metric_key(resolved)
    if key is not None:
        pending_by_id.pop(key, None)
    discard_pending_indexes(resolved, pending_by_name_cycle, pending_by_cycle)
    if resolved.name != xys_name:
        diagnostics.name_mismatches += 1
    if resolved.cycles != xys0:
        diagnostics.cycle_mismatches += 1
    resolved.name = xys_name
    resolved.xys0 = xys0
    resolved.xys1 = xys1
    return resolved


def discard_pending_indexes(metric: Metric, pending_by_name_cycle, pending_by_cycle) -> None:
    name_queue = pending_by_name_cycle.get((metric.name, metric.cycles))
    if name_queue:
        for index, candidate in enumerate(name_queue):
            if candidate is metric:
                del name_queue[index]
                break
        if not name_queue:
            del pending_by_name_cycle[(metric.name, metric.cycles)]
    cycle_queue = pending_by_cycle.get(metric.cycles)
    if cycle_queue:
        for index, candidate in enumerate(cycle_queue):
            if candidate is metric:
                del cycle_queue[index]
                break
        if not cycle_queue:
            del pending_by_cycle[metric.cycles]


def get_kernel_launches(path: Path) -> tuple[list[Metric], Diagnostics]:
    results: list[Metric] = []
    diagnostics = Diagnostics()
    pending_by_id: dict[tuple[int, int], Metric] = {}
    pending_xys_by_id: dict[tuple[int, int], tuple[str, int, int]] = {}
    pending_by_name_cycle: dict[tuple[str, int], deque[Metric]] = defaultdict(deque)
    pending_by_cycle: dict[int, deque[Metric]] = defaultdict(deque)
    active_name: str | None = None
    active_body: list[bytes] = []

    with path.open("rb", buffering=4 * 1024 * 1024) as source:
        for line in source:
            if b"xys0_cycle:" in line:
                xys = parse_xys(line)
                if xys is not None:
                    batch_id, kernel_index, xys_name, _, xys0, xys1 = xys
                    candidate = None
                    if batch_id is not None and kernel_index is not None:
                        key = (batch_id, kernel_index)
                        candidate = pending_by_id.get(key)
                        if candidate is None:
                            pending_xys_by_id[key] = (xys_name, xys0, xys1)
                            continue
                    if candidate is None:
                        candidate = Metric(xys_name, "", xys0, 0, 0, 0)
                    if batch_id is not None:
                        candidate.batch_id = batch_id
                    if kernel_index is not None:
                        candidate.kernel_index = kernel_index
                    resolved = resolve_metric(
                        candidate, xys_name, xys0, xys1, pending_by_id,
                        pending_by_name_cycle, pending_by_cycle, diagnostics,
                    )
                    if resolved is not None:
                        results.append(resolved)
                continue

            launch = LAUNCH.search(line) if b" launch" in line else None
            cycle_match = None
            if b"cycles:" in line:
                cycle_match = CYCLE_ID.search(line) or CYCLE.search(line)

            if launch is not None:
                if active_name is not None:
                    diagnostics.unprofiled_launches += 1
                active_name = launch.group(1).decode()
                active_body = [
                    line[:cycle_match.start()] if cycle_match is not None else line
                ]

            identified = CYCLE_ID.search(line) if cycle_match is not None else None
            if cycle_match is not None and (active_name is not None or identified is not None):
                identified_name = (
                    identified.group(3).decode() if identified is not None else None
                )
                has_body = active_name is not None and (
                    identified_name is None or identified_name == active_name
                )
                body = (
                    remove_ansi(b"".join(active_body).decode(errors="replace"))
                    if has_body
                    else ""
                )
                parsed = parse_cycle(line, active_name or identified_name or "", body)
                if parsed is not None:
                    diagnostics.parsed_events += 1
                    diagnostics.total_cycles += parsed.cycles
                    key = metric_key(parsed)
                    early_xys = pending_xys_by_id.pop(key, None) if key else None
                    if early_xys is not None:
                        xys_name, xys0, xys1 = early_xys
                        diagnostics.id_events += 1
                        if parsed.name != xys_name:
                            diagnostics.name_mismatches += 1
                        if parsed.cycles != xys0:
                            diagnostics.cycle_mismatches += 1
                        parsed.name = xys_name
                        parsed.xys0 = xys0
                        parsed.xys1 = xys1
                        results.append(parsed)
                    else:
                        if key is not None:
                            pending_by_id[key] = parsed
                        pending_by_name_cycle[(parsed.name, parsed.cycles)].append(parsed)
                        pending_by_cycle[parsed.cycles].append(parsed)
                if has_body:
                    active_name = None
                    active_body = []
                continue

            if CYCLE_MARKER.match(line):
                if active_name is not None:
                    diagnostics.unprofiled_launches += 1
                active_name = None
                active_body = []
                continue

            if active_name is not None:
                if launch is None:
                    active_body.append(line)

    if active_name is not None:
        diagnostics.unprofiled_launches += 1
    diagnostics.unmatched_cycles = sum(len(queue) for queue in pending_by_cycle.values())
    if not results:
        raise RuntimeError(f"No cycle records found in {path}")
    return results, diagnostics


def aggregate(metrics: list[Metric], key_function) -> dict:
    groups = {}
    for metric in metrics:
        key = key_function(metric)
        if key not in groups:
            groups[key] = Metric(metric.name, metric.body, 0, 0, 0, 0, n=0)
        groups[key] += metric
    return groups


def format_report(
    metrics: list[Metric],
    diagnostics: Diagnostics,
    kernel_filter: str | None = None,
    metadata: list[str] | None = None,
) -> str:
    total = Metric("Total", "", 0, 0, 0, 0, n=0)
    for metric in metrics:
        total += metric
    by_body = aggregate(metrics, lambda metric: (metric.name, metric.body))
    by_name = aggregate(metrics, lambda metric: metric.name)
    lines = ["# parser=batch-aware-v2"]
    if metadata:
        lines.extend(f"# {line}" for line in metadata)
    lines.extend([
        f"# events={len(metrics)} id_events={diagnostics.id_events} "
        f"fallback_events={diagnostics.fallback_events} "
        f"unmatched_xys={diagnostics.unmatched_xys} "
        f"unmatched_cycles={diagnostics.unmatched_cycles} "
        f"unprofiled_launches={diagnostics.unprofiled_launches} "
        f"name_mismatches={diagnostics.name_mismatches} "
        f"cycle_mismatches={diagnostics.cycle_mismatches}",
        "",
    ])
    for metric in sorted(by_body.values(), key=lambda item: item.cycles, reverse=True):
        if kernel_filter is not None and metric.name != kernel_filter:
            continue
        lines.extend(["─" * 122, metric.format(total), metric.body])
    lines.append("")
    for metric in sorted(by_name.values(), key=lambda item: item.cycles, reverse=True):
        lines.append(metric.format(total))
    lines.extend(["", total.format(total)])
    return "\n".join(lines) + "\n"


def diagnostics_for_metrics(metrics: list[Metric]) -> Diagnostics:
    return Diagnostics(
        total_cycles=sum(metric.cycles for metric in metrics),
        parsed_events=len(metrics),
        id_events=sum(metric_key(metric) is not None for metric in metrics),
        fallback_events=sum(metric_key(metric) is None for metric in metrics),
    )


def split_forward_rounds(metrics: list[Metric], boundary: str) -> list[ForwardRound]:
    starts = [index for index, metric in enumerate(metrics) if metric.name == boundary]
    if not starts:
        raise RuntimeError(
            f"Forward boundary kernel {boundary!r} was not found; "
            "choose one launch per forward with --forward-boundary"
        )

    rounds: list[ForwardRound] = []
    if starts[0] > 0:
        rounds.append(ForwardRound(0, "preamble", metrics[: starts[0]]))
    for number, start in enumerate(starts, 1):
        end = starts[number] if number < len(starts) else len(metrics)
        rounds.append(ForwardRound(number, f"round_{number:04d}", metrics[start:end]))
    return rounds


def format_forward_summary(
    rounds: list[ForwardRound], boundary: str, top_n: int
) -> str:
    forward_count = sum(round_.number > 0 for round_ in rounds)
    lines = [
        f"# forward boundary: {boundary}",
        f"# forwards: {forward_count}",
        "# note: each forward starts at the boundary kernel; the final forward may be incomplete",
        "",
        f"{'round':<14} {'events':>8} {'cycles':>14} {'time_ms':>12}  top operators",
    ]
    for round_ in rounds:
        total_cycles = sum(metric.cycles for metric in round_.metrics)
        by_name = aggregate(round_.metrics, lambda metric: metric.name)
        top = sorted(by_name.values(), key=lambda metric: metric.cycles, reverse=True)
        top = top[: max(top_n, 0)]
        top_text = ", ".join(
            f"{metric.name}={metric.cycles / max(total_cycles, 1) * 100:.2f}%/"
            f"{metric.cycles / FREQUENCY_MHZ / 1000:.2f}ms"
            for metric in top
        )
        lines.append(
            f"{round_.name:<14} {len(round_.metrics):>8} {total_cycles:>14} "
            f"{total_cycles / FREQUENCY_MHZ / 1000:>12.2f}  {top_text}"
        )
    return "\n".join(lines) + "\n"


def write_forward_comparison_csv(
    path: Path, rounds: list[ForwardRound]
) -> None:
    with path.open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(
            [
                "round",
                "label",
                "events",
                "total_cycles",
                "total_time_ms",
                "kernel",
                "count",
                "cycles",
                "time_ms",
                "share_pct",
            ]
        )
        for round_ in rounds:
            total_cycles = sum(metric.cycles for metric in round_.metrics)
            by_name = aggregate(round_.metrics, lambda metric: metric.name)
            for metric in sorted(
                by_name.values(), key=lambda item: item.cycles, reverse=True
            ):
                writer.writerow(
                    [
                        round_.number,
                        round_.name,
                        len(round_.metrics),
                        total_cycles,
                        f"{total_cycles / FREQUENCY_MHZ / 1000:.6f}",
                        metric.name,
                        metric.count,
                        metric.cycles,
                        f"{metric.cycles / FREQUENCY_MHZ / 1000:.6f}",
                        f"{metric.cycles / max(total_cycles, 1) * 100:.6f}",
                    ]
                )


def write_forward_reports(
    input_path: Path,
    metrics: list[Metric],
    output_dir: Path,
    boundary: str,
    kernel_filter: str | None,
    top_n: int,
) -> None:
    rounds = split_forward_rounds(metrics, boundary)
    output_dir.mkdir(parents=True, exist_ok=True)
    for round_ in rounds:
        metadata = [
            f"source={input_path.name}",
            f"forward_boundary={boundary}",
            f"round={round_.name}",
        ]
        report = format_report(
            round_.metrics,
            diagnostics_for_metrics(round_.metrics),
            kernel_filter,
            metadata,
        )
        (output_dir / f"{round_.name}_kernels.txt").write_text(
            report
        )
    (output_dir / "summary.txt").write_text(
        format_forward_summary(rounds, boundary, top_n)
    )
    write_forward_comparison_csv(output_dir / "comparison.csv", rounds)
    print(
        f"  split -> {output_dir} "
        f"forwards={sum(round_.number > 0 for round_ in rounds)}"
    )


def process_log(
    input_path: Path,
    output_path: Path,
    kernel_filter: str | None = None,
    split_forward: bool = False,
    forward_boundary: str = "custom_embedding_dense",
    summary_top: int = 5,
) -> None:
    print(f"{input_path} -> {output_path}")
    metrics, diagnostics = get_kernel_launches(input_path)
    output_path.write_text(format_report(metrics, diagnostics, kernel_filter))
    if split_forward:
        write_forward_reports(
            input_path,
            metrics,
            forward_output_dir(input_path, kernel_filter),
            forward_boundary,
            kernel_filter,
            summary_top,
        )
    print(
        f"events={len(metrics)} id={diagnostics.id_events} fallback={diagnostics.fallback_events} "
        f"unmatched_xys={diagnostics.unmatched_xys} unmatched_cycles={diagnostics.unmatched_cycles} "
        f"unprofiled={diagnostics.unprofiled_launches}"
    )


def process_trace(path: Path) -> None:
    with path.open("r", errors="ignore") as source:
        trace = json.load(source)
    timestamps = [event["ts"] for event in trace.get("traceEvents", []) if "ts" in event]
    if timestamps:
        minimum = min(timestamps)
        for event in trace["traceEvents"]:
            if "ts" in event:
                event["ts"] -= minimum
    with path.open("w") as destination:
        json.dump(trace, destination, indent=4)


def output_path(path: Path, kernel_filter: str | None = None) -> Path:
    suffix = f"_{kernel_filter}.txt" if kernel_filter is not None else "_kernels.txt"
    if path.suffix:
        return path.with_suffix("").with_name(path.stem + suffix)
    return path.with_name(path.name + suffix)


def forward_output_dir(path: Path, kernel_filter: str | None = None) -> Path:
    suffix = f"_forwards_{kernel_filter}" if kernel_filter is not None else "_forwards"
    if path.suffix:
        return path.with_suffix("").with_name(path.stem + suffix)
    return path.with_name(path.name + suffix)


def process_path(
    path: Path,
    kernel_filter: str | None = None,
    split_forward: bool = False,
    forward_boundary: str = "custom_embedding_dense",
    summary_top: int = 5,
) -> None:
    if path.is_dir():
        for root, _, files in os.walk(path):
            for filename in files:
                current = Path(root) / filename
                if current.suffix in {".log", ".ansi"}:
                    process_log(
                        current,
                        output_path(current, kernel_filter),
                        kernel_filter,
                        split_forward,
                        forward_boundary,
                        summary_top,
                    )
                elif current.suffix == ".json":
                    process_trace(current)
    else:
        process_log(
            path,
            output_path(path, kernel_filter),
            kernel_filter,
            split_forward,
            forward_boundary,
            summary_top,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-aware Synapse kernel cycle parser")
    parser.add_argument("path", type=Path)
    parser.add_argument("--kernel", help="Filter to show only specific kernel details")
    parser.add_argument(
        "--split-forward",
        action="store_true",
        help="Additionally write one cycle report per model forward",
    )
    parser.add_argument(
        "--forward-boundary",
        default="custom_embedding_dense",
        help="Kernel name that marks the start of each forward (default: %(default)s)",
    )
    parser.add_argument(
        "--summary-top",
        type=int,
        default=5,
        help="Number of top kernels to include per forward in summary.txt (default: 5)",
    )
    args = parser.parse_args()
    if not args.path.exists():
        parser.error(f"file not found: {args.path}")
    process_path(
        args.path,
        args.kernel,
        args.split_forward,
        args.forward_boundary,
        args.summary_top,
    )


if __name__ == "__main__":
    main()
