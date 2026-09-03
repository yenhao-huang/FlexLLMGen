"""Search reproducible fine-grained Qwen weight-home percentages.

Every measurement runs in a new process.  Child stdout/stderr, successful
benchmark JSON, failures, the environment snapshot, aggregate statistics, and
the final selection are retained under the experiment directory.
"""

import argparse
import datetime as datetime_module
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from flexllmgen.hf_backend import append_jsonl
from flexllmgen.qwen_flex import FlexQwenPolicy, build_qwen_plan


DEFAULT_WEIGHTS = (15, 20, 30, 33, 40, 45, 49, 50)
GIB = 1 << 30


def _gpu_snapshot() -> List[Mapping[str, object]]:
    output = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = []
    for line in output.splitlines():
        index, name, total, used, free = [item.strip() for item in line.split(",")]
        rows.append({
            "index": int(index),
            "name": name,
            "memory_total_mib": int(total),
            "memory_used_mib": int(used),
            "memory_free_mib": int(free),
        })
    return rows


def _cpu_available_bytes() -> int:
    with open("/proc/meminfo", "r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _read_jsonl(path: Path) -> List[Mapping[str, object]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _command(args, weight_percent: int, output_path: Path) -> List[str]:
    return [
        sys.executable,
        "-m",
        "flexllmgen.hf_opt",
        "--model",
        args.model,
        "--quantization",
        "int8-torchao",
        "--gpu-memory",
        *args.gpu_memory,
        "--cpu-memory",
        args.cpu_memory,
        "--offload-dir",
        args.offload_dir,
        "--flex-percent",
        str(weight_percent),
        str(100 - weight_percent),
        "100",
        "0",
        "100",
        "0",
        "--flex-compute-device",
        str(args.compute_device),
        "--local-files-only",
        "--batch-size",
        str(args.batch_size),
        "--prompt",
        args.prompt,
        "--warmup-tokens",
        str(args.warmup_tokens),
        "--gen-len",
        str(args.gen_len),
        "--output-jsonl",
        str(output_path),
    ]


def _completed_keys(rows: Iterable[Mapping[str, object]]):
    return {
        (int(row["candidate_weight_gpu_percent"]), int(row["repeat"]))
        for row in rows
        if "candidate_weight_gpu_percent" in row and "repeat" in row
    }


def _median(values: Sequence[float]):
    return statistics.median(values) if values else None


def _failure_reason(row: Mapping[str, object]) -> str:
    text = str(row.get("error_tail", ""))
    if "CUDA out of memory" in text or "torch.OutOfMemoryError" in text:
        return "cuda_oom"
    if row.get("returncode") is not None:
        return "returncode_{}".format(row["returncode"])
    return "unknown"


def aggregate_rows(
    rows: Sequence[Mapping[str, object]],
    candidates: Sequence[int],
    repeats: int,
    gpu_budget_gib: float,
    required_headroom_gib: float,
) -> List[Mapping[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        if "candidate_weight_gpu_percent" in row:
            grouped[int(row["candidate_weight_gpu_percent"])].append(row)

    aggregates = []
    eligible_peak = (gpu_budget_gib - required_headroom_gib) * GIB
    for candidate in candidates:
        candidate_rows = grouped[candidate]
        successes = [row for row in candidate_rows if row.get("status") == "success"]
        failures = [row for row in candidate_rows if row.get("status") != "success"]
        throughputs = [float(row["tokens_per_second"]) for row in successes]
        elapsed = [float(row["elapsed_seconds"]) for row in successes]
        peaks = [int(row["gpu_peak_memory_bytes"][str(row["compute_device"])]) for row in successes]
        median_tps = _median(throughputs)
        aggregate = {
            "weight_gpu_percent": candidate,
            "weight_cpu_percent": 100 - candidate,
            "required_repeats": repeats,
            "successes": len(successes),
            "failures": len(failures),
            "failure_reasons": dict(Counter(_failure_reason(row) for row in failures)),
            "median_tokens_per_second": median_tps,
            "min_tokens_per_second": min(throughputs) if throughputs else None,
            "max_tokens_per_second": max(throughputs) if throughputs else None,
            "median_elapsed_seconds": _median(elapsed),
            "max_gpu_peak_memory_bytes": max(peaks) if peaks else None,
            "max_gpu_peak_memory_gib": max(peaks) / GIB if peaks else None,
            "headroom_from_budget_gib": (
                gpu_budget_gib - max(peaks) / GIB if peaks else None
            ),
            "stable": len(successes) == repeats and not failures,
            "eligible": (
                len(successes) == repeats
                and not failures
                and max(peaks, default=eligible_peak + 1) <= eligible_peak
            ),
        }
        aggregates.append(aggregate)
    return aggregates


def select_best(aggregates: Sequence[Mapping[str, object]]):
    eligible = [row for row in aggregates if row["eligible"]]
    if not eligible:
        return None
    # Throughput is primary. Lower GPU peak, then lower GPU percentage, break
    # ties deterministically and preserve more operational headroom.
    return max(eligible, key=lambda row: (
        row["median_tokens_per_second"],
        -row["max_gpu_peak_memory_bytes"],
        -row["weight_gpu_percent"],
    ))


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _write_report(path: Path, manifest, aggregates, selected) -> None:
    lines = [
        "# Qwen3.8-27B fine-grained weight percentage selection",
        "",
        "## Fixed benchmark parameters",
        "",
        "- Model: `{}`".format(manifest["model"]),
        "- Weight GPU candidates: `{}`".format(
            ", ".join(str(value) for value in manifest["candidates_weight_gpu_percent"])
        ),
        "- Repeats per candidate: {}".format(manifest["repeats"]),
        "- Isolation: every repeat ran in a new Python process",
        "- Python: `{}`".format(manifest["python"]),
        "",
        "```json",
        json.dumps(manifest["benchmark"], indent=2, sort_keys=True),
        "```",
        "",
        "## Results",
        "",
        "| GPU/CPU weight % | Success | Median tok/s | Range tok/s | Peak GPU GiB | Budget headroom GiB | Eligible |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in aggregates:
        median = "-" if row["median_tokens_per_second"] is None else "{:.6f}".format(row["median_tokens_per_second"])
        value_range = "-" if row["min_tokens_per_second"] is None else "{:.6f}–{:.6f}".format(
            row["min_tokens_per_second"], row["max_tokens_per_second"]
        )
        peak = "-" if row["max_gpu_peak_memory_gib"] is None else "{:.3f}".format(row["max_gpu_peak_memory_gib"])
        headroom = "-" if row["headroom_from_budget_gib"] is None else "{:.3f}".format(row["headroom_from_budget_gib"])
        lines.append(
            "| {}/{} | {}/{} | {} | {} | {} | {} | {} |".format(
                row["weight_gpu_percent"],
                row["weight_cpu_percent"],
                row["successes"],
                row["required_repeats"],
                median,
                value_range,
                peak,
                headroom,
                "yes" if row["eligible"] else "no",
            )
        )
    lines.extend(["", "## Selected ratio", ""])
    if selected is None:
        lines.append("No candidate satisfied the stability and GPU-headroom criteria.")
    else:
        baseline_30 = next((row for row in aggregates if row["weight_gpu_percent"] == 30), None)
        conservative_45 = next((row for row in aggregates if row["weight_gpu_percent"] == 45), None)
        selected_plan = manifest["plans"][str(selected["weight_gpu_percent"])]
        lines.extend([
            "Selected **{}% GPU / {}% CPU** for weights.".format(
                selected["weight_gpu_percent"], selected["weight_cpu_percent"]
            ),
            "",
            "Selection rule: highest median measured throughput among candidates that completed all repeats "
            "and retained at least {} GiB below the {} GiB PyTorch allocation budget.".format(
                manifest["selection"]["required_headroom_gib"],
                manifest["selection"]["gpu_budget_gib"],
            ),
            "",
            "- Median throughput: {:.6f} token/s".format(selected["median_tokens_per_second"]),
            "- Worst observed GPU peak: {:.3f} GiB".format(selected["max_gpu_peak_memory_gib"]),
            "- Remaining budget headroom: {:.3f} GiB".format(selected["headroom_from_budget_gib"]),
            "- Checkpoint logical bytes by home: `{}`".format(
                json.dumps(selected_plan["checkpoint_bytes_by_home"], sort_keys=True)
            ),
        ])
        if baseline_30 and baseline_30["median_tokens_per_second"]:
            improvement = (selected["median_tokens_per_second"] / baseline_30["median_tokens_per_second"] - 1) * 100
            lines.append("- Throughput improvement over 30/70: {:.2f}%".format(improvement))
        if conservative_45 and conservative_45["median_tokens_per_second"]:
            improvement = (selected["median_tokens_per_second"] / conservative_45["median_tokens_per_second"] - 1) * 100
            lines.append("- Throughput improvement over 45/55: {:.2f}%".format(improvement))
        reproduction_command = " ".join([
            "python -m flexllmgen.hf_opt",
            "--model {}".format(manifest["model"]),
            "--quantization int8-torchao",
            "--gpu-memory {}".format(" ".join(manifest["benchmark"]["gpu_memory"])),
            "--cpu-memory {}".format(manifest["benchmark"]["cpu_memory"]),
            "--flex-percent {} {} 100 0 100 0".format(
                selected["weight_gpu_percent"], selected["weight_cpu_percent"]
            ),
            "--flex-compute-device {}".format(manifest["benchmark"]["compute_device"]),
            "--offload-dir /data/flexllmgen-offload/qwen3.8-27b-fine",
            "--local-files-only",
            "--batch-size {}".format(manifest["benchmark"]["batch_size"]),
            "--warmup-tokens {}".format(manifest["benchmark"]["warmup_tokens"]),
            "--gen-len {}".format(manifest["benchmark"]["gen_len"]),
            "--prompt '{}'".format(manifest["benchmark"]["prompt"]),
            "--output-jsonl exp/qwen3_8_27b/selected_weight_results.jsonl",
        ])
        lines.extend([
            "",
            "### Reproduction command",
            "",
            "```bash",
            "source /data/flexllmgen-qwen-venv/bin/activate",
            "cd /data/FlexLLMGen",
            reproduction_command,
            "```",
        ])
    failed = [row for row in aggregates if row["failures"]]
    if failed:
        lines.extend(["", "## Failures", ""])
        for row in failed:
            lines.append(
                "- {}/{} failed {}/{} repeats: `{}`.".format(
                    row["weight_gpu_percent"],
                    row["weight_cpu_percent"],
                    row["failures"],
                    row["required_repeats"],
                    json.dumps(row["failure_reasons"], sort_keys=True),
                )
            )
    lines.extend([
        "",
        "Raw success and failure records are in `weight_search.jsonl`; per-run stdout/stderr is in `weight_search_runs/`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/Qwen-Qwen3.8-27B")
    parser.add_argument("--weight-percent", type=int, nargs="+", default=list(DEFAULT_WEIGHTS))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--gpu-memory", nargs="+", default=["1GiB", "15GiB"])
    parser.add_argument("--cpu-memory", default="35GiB")
    parser.add_argument("--compute-device", type=int, default=1)
    parser.add_argument("--gpu-budget-gib", type=float, default=15.0)
    parser.add_argument("--required-headroom-gib", type=float, default=1.0)
    parser.add_argument("--offload-dir", default="/data/flexllmgen-offload/qwen3.8-27b-fine-search")
    parser.add_argument("--prompt", default="Explain tensor offloading in one sentence.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-tokens", type=int, default=1)
    parser.add_argument("--gen-len", type=int, default=4)
    parser.add_argument("--result-dir", default="exp/qwen3_8_27b")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    candidates = sorted(set(args.weight_percent))
    if any(value < 0 or value > 100 for value in candidates):
        raise SystemExit("weight percentages must be between 0 and 100")
    if args.required_headroom_gib < 0 or args.required_headroom_gib >= args.gpu_budget_gib:
        raise SystemExit("required headroom must be non-negative and smaller than the GPU budget")

    result_dir = Path(args.result_dir).resolve()
    results_path = result_dir / "weight_search.jsonl"
    manifest_path = result_dir / "weight_search_manifest.json"
    aggregate_path = result_dir / "weight_search_aggregate.json"
    selection_path = result_dir / "selected_weight_ratio.json"
    report_path = result_dir / "WEIGHT_SELECTION.md"
    run_dir = result_dir / "weight_search_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    plans = {}
    for candidate in candidates:
        plan = build_qwen_plan(
            args.model,
            FlexQwenPolicy.from_sequence([candidate, 100 - candidate, 100, 0, 100, 0]),
            args.compute_device,
        )
        plans[str(candidate)] = {
            "stage_count": len(plan.stages),
            "checkpoint_bytes_by_home": plan.totals_by_home(),
            "device_map_entries": len(plan.device_map),
        }

    manifest = {
        "created_at_utc": datetime_module.datetime.now(datetime_module.timezone.utc).isoformat(),
        "python": sys.executable,
        "model": args.model,
        "candidates_weight_gpu_percent": candidates,
        "repeats": args.repeats,
        "initial_gpu_snapshot": _gpu_snapshot(),
        "initial_cpu_available_bytes": _cpu_available_bytes(),
        "benchmark": {
            "quantization": "int8-torchao",
            "gpu_memory": args.gpu_memory,
            "cpu_memory": args.cpu_memory,
            "compute_device": args.compute_device,
            "cache_gpu_percent": 100,
            "cache_cpu_percent": 0,
            "activation_gpu_percent": 100,
            "activation_cpu_percent": 0,
            "batch_size": args.batch_size,
            "prompt": args.prompt,
            "warmup_tokens": args.warmup_tokens,
            "gen_len": args.gen_len,
            "local_files_only": True,
        },
        "selection": {
            "primary_metric": "median_tokens_per_second",
            "requires_all_repeats_successful": True,
            "gpu_budget_gib": args.gpu_budget_gib,
            "required_headroom_gib": args.required_headroom_gib,
            "tie_breakers": ["lower_max_gpu_peak_memory_bytes", "lower_weight_gpu_percent"],
        },
        "plans": plans,
    }
    _write_json(manifest_path, manifest)

    existing = _read_jsonl(results_path) if args.resume else []
    completed = _completed_keys(existing)
    if results_path.exists() and not args.resume:
        raise SystemExit("{} already exists; pass --resume or choose another --result-dir".format(results_path))

    for candidate in candidates:
        for repeat in range(1, args.repeats + 1):
            if (candidate, repeat) in completed:
                print("Skipping completed weight={} repeat={}".format(candidate, repeat), flush=True)
                continue
            stamp = time.time_ns()
            prefix = "weight-{:03d}-repeat-{:02d}-{}".format(candidate, repeat, stamp)
            child_result = run_dir / (prefix + ".jsonl")
            log_path = run_dir / (prefix + ".log")
            command = _command(args, candidate, child_result)
            if args.dry_run:
                print(json.dumps({"candidate": candidate, "repeat": repeat, "command": command}))
                continue
            print("Running weight={} repeat={}...".format(candidate, repeat), flush=True)
            before_gpu = _gpu_snapshot()
            before_cpu = _cpu_available_bytes()
            started = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as log:
                completed_process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
            wall_seconds = time.perf_counter() - started
            child_rows = _read_jsonl(child_result)
            common = {
                "candidate_weight_gpu_percent": candidate,
                "candidate_weight_cpu_percent": 100 - candidate,
                "repeat": repeat,
                "compute_device": args.compute_device,
                "process_wall_seconds": wall_seconds,
                "returncode": completed_process.returncode,
                "command": command,
                "log_path": str(log_path),
                "gpu_snapshot_before": before_gpu,
                "cpu_available_bytes_before": before_cpu,
                "plan": plans[str(candidate)],
            }
            if completed_process.returncode == 0 and child_rows:
                record = dict(child_rows[-1])
                record.update(common)
                record["status"] = "success"
            else:
                tail = ""
                with log_path.open("r", encoding="utf-8", errors="replace") as log:
                    tail = log.read()[-12000:]
                record = dict(common)
                record.update({
                    "status": "failed",
                    "error_tail": tail,
                })
            append_jsonl(str(results_path), record)

    if args.dry_run:
        return 0
    rows = _read_jsonl(results_path)
    aggregates = aggregate_rows(
        rows,
        candidates,
        args.repeats,
        args.gpu_budget_gib,
        args.required_headroom_gib,
    )
    selected = select_best(aggregates)
    _write_json(aggregate_path, aggregates)
    selected_policy = None if selected is None else [
        selected["weight_gpu_percent"], selected["weight_cpu_percent"], 100, 0, 100, 0
    ]
    _write_json(selection_path, {
        "selected": selected,
        "selected_flex_percent": selected_policy,
        "fixed_benchmark_parameters": manifest["benchmark"],
        "selection_criteria": manifest["selection"],
        "source_results": str(results_path),
    })
    _write_report(report_path, manifest, aggregates, selected)
    print(json.dumps({"selected": selected, "report": str(report_path)}, indent=2, sort_keys=True))
    return 0 if selected is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
