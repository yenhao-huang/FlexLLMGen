"""Run the reproducible Qwen3.8 tensor-placement search.

Each candidate runs in a fresh process so CUDA allocations and Accelerate
hooks from one placement cannot affect the next.  Successful benchmark rows
and failures are both appended to JSONL for auditability and resumption.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import List, Sequence

from flexllmgen.hf_backend import append_jsonl
from flexllmgen.placement import PlacementCandidate, generate_candidates


MODEL_BYTES = 55_562_855_904


def _gpu_free_gib() -> List[int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [max(1, int(line.strip()) // 1024) for line in output.splitlines() if line.strip()]


def _cpu_available_gib() -> int:
    with open("/proc/meminfo", "r", encoding="utf-8") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return max(1, int(line.split()[1]) // (1024 * 1024))
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def command_for_candidate(args, candidate: PlacementCandidate) -> List[str]:
    is_dram_baseline = candidate.strategy == "cpu"
    baseline_is_torchao = is_dram_baseline and args.quantization == "int8-torchao"
    quantization = args.quantization
    dtype = "auto"
    if is_dram_baseline and not baseline_is_torchao:
        # bitsandbytes does not quantize a CPU-only map, so its baseline falls
        # back to the honest BF16 reference instead of being mislabeled int8.
        quantization = "none"
        dtype = "bfloat16"
    command = [
        sys.executable,
        "-m",
        "flexllmgen.hf_opt",
        "--model",
        args.model,
        "--quantization",
        quantization,
        "--device-map",
        candidate.strategy,
        "--dtype",
        dtype,
        "--cpu-memory",
        "{}GiB".format(candidate.cpu_memory_gib),
        "--offload-dir",
        args.offload_dir,
        "--prompt",
        args.prompt,
        "--batch-size",
        str(args.batch_size),
        "--gen-len",
        str(args.gen_len),
        "--warmup-tokens",
        str(args.warmup_tokens),
        "--output-jsonl",
        args.output_jsonl,
    ]
    if not is_dram_baseline:
        command.extend([
            "--gpu-memory",
            *["{}GiB".format(value) for value in candidate.gpu_memory_gib],
        ])
    if args.local_files_only:
        command.append("--local-files-only")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/Qwen-Qwen3.8-27B")
    parser.add_argument(
        "--quantization",
        choices=("none", "int8", "int8-torchao"),
        default="int8-torchao",
    )
    parser.add_argument("--offload-dir", default="/data/flexllmgen-offload/qwen3.8-27b")
    parser.add_argument("--output-jsonl", default="exp/qwen3_8_27b/results.jsonl")
    parser.add_argument("--prompt", default="Explain tensor offloading in one sentence.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gen-len", type=int, default=16)
    parser.add_argument("--warmup-tokens", type=int, default=1)
    parser.add_argument("--gpu-free-gib", type=int, nargs="+")
    parser.add_argument("--cpu-available-gib", type=int)
    parser.add_argument("--candidate", action="append", help="Run only a named candidate.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] = None) -> int:
    args = build_parser().parse_args(argv)
    gpu_free = args.gpu_free_gib or _gpu_free_gib()
    cpu_available = args.cpu_available_gib or _cpu_available_gib()
    candidates = generate_candidates(gpu_free, cpu_available)
    if args.candidate:
        selected = set(args.candidate)
        candidates = [candidate for candidate in candidates if candidate.name in selected]
        missing = selected.difference(candidate.name for candidate in candidates)
        if missing:
            raise SystemExit("unknown candidate(s): {}".format(", ".join(sorted(missing))))

    manifest = {
        "model": args.model,
        "model_bytes": MODEL_BYTES,
        "candidate_quantization": args.quantization,
        "dram_baseline_quantization": (
            "int8-torchao" if args.quantization == "int8-torchao" else "none"
        ),
        "dram_baseline_dtype": (
            "auto" if args.quantization == "int8-torchao" else "bfloat16"
        ),
        "gpu_free_gib": gpu_free,
        "cpu_available_gib": cpu_available,
        "candidates": [],
    }
    for candidate in candidates:
        row = dict(candidate.to_dict())
        is_torchao_baseline = (
            candidate.strategy == "cpu" and args.quantization == "int8-torchao"
        )
        row["quantization"] = (
            "int8-torchao" if is_torchao_baseline
            else "none (bfloat16)" if candidate.strategy == "cpu"
            else args.quantization
        )
        row["estimated_weight_placement"] = dict(
            candidate.estimated_weight_placement(
                MODEL_BYTES,
                bits=(
                    8 if is_torchao_baseline
                    else 16 if candidate.strategy == "cpu"
                    else 8 if args.quantization in {"int8", "int8-torchao"}
                    else 16
                ),
            )
        )
        row["command"] = command_for_candidate(args, candidate)
        manifest["candidates"].append(row)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    failures = 0
    for candidate in candidates:
        print("Running {}...".format(candidate.name), flush=True)
        completed = subprocess.run(command_for_candidate(args, candidate), text=True)
        if completed.returncode:
            failures += 1
            append_jsonl(args.output_jsonl, {
                "candidate": candidate.name,
                "status": "failed",
                "returncode": completed.returncode,
            })
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
