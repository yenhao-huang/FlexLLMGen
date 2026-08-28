"""CLI for Hugging Face-backed FlexLLMGen inference.

Example:
    python -m flexllmgen.hf_opt \
      --model /models/Qwen-Qwen3.8-27B --quantization int8 \
      --device-map balanced --gpu-memory 18GiB 18GiB \
      --cpu-memory 36GiB --local-files-only
"""

import argparse
import json
from typing import Dict, Optional, Sequence, Union

from flexllmgen.hf_backend import HFOffloadConfig, HFOffloadLM, append_jsonl
from flexllmgen.qwen_flex import (
    FlexQwenPolicy,
    build_qwen_plan,
)


def _max_memory(gpu_memory: Optional[Sequence[str]], cpu_memory: Optional[str]):
    if gpu_memory is None and cpu_memory is None:
        return None
    result: Dict[Union[int, str], str] = {}
    for index, value in enumerate(gpu_memory or []):
        result[index] = value
    if cpu_memory is not None:
        result["cpu"] = cpu_memory
    return result


def _size_bytes(value: str) -> int:
    units = {
        "b": 1,
        "kb": 1000,
        "mb": 1000 ** 2,
        "gb": 1000 ** 3,
        "tb": 1000 ** 4,
        "kib": 1 << 10,
        "mib": 1 << 20,
        "gib": 1 << 30,
        "tib": 1 << 40,
    }
    normalized = value.strip().lower()
    for suffix in sorted(units, key=len, reverse=True):
        if normalized.endswith(suffix):
            return int(float(normalized[:-len(suffix)]) * units[suffix])
    raise ValueError("memory size must include a byte unit: {}".format(value))


def _compute_device(gpu_memory: Optional[Sequence[str]], requested: Optional[int]) -> int:
    if requested is not None:
        return requested
    if not gpu_memory:
        return 0
    return max(range(len(gpu_memory)), key=lambda index: _size_bytes(gpu_memory[index]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run modern Hugging Face models with FlexLLMGen placement controls."
    )
    parser.add_argument("--model", default="/models/Qwen-Qwen3.8-27B")
    parser.add_argument(
        "--quantization",
        choices=("none", "int8", "int8-torchao"),
        default="int8-torchao",
    )
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0", "sequential", "cpu"),
        default="auto",
    )
    parser.add_argument("--gpu-memory", nargs="+", metavar="SIZE")
    parser.add_argument("--cpu-memory", metavar="SIZE")
    parser.add_argument("--offload-dir", default="/data/flexllmgen-offload/qwen3.8-27b")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-cpu-offload", action="store_true")
    parser.add_argument("--prompt", default="Explain tensor offloading in one sentence.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gen-len", type=int, default=16)
    parser.add_argument("--warmup-tokens", type=int, default=1)
    parser.add_argument("--output-jsonl")
    parser.add_argument(
        "--flex-percent",
        nargs=6,
        type=float,
        metavar=("W_GPU", "W_CPU", "CACHE_GPU", "CACHE_CPU", "ACT_GPU", "ACT_CPU"),
        help=(
            "Enable Qwen fine-grained FlexLLMGen placement. The six values are "
            "weight GPU/CPU, cache GPU/CPU, and activation GPU/CPU percentages; "
            "each remaining percentage is assigned to disk."
        ),
    )
    parser.add_argument(
        "--flex-compute-device",
        type=int,
        help="CUDA index used for streamed projection/LN/FFN execution (defaults to the GPU with the largest budget).",
    )
    parser.add_argument(
        "--print-flex-plan",
        action="store_true",
        help="Include every fine-grained Qwen stage in dry-run output.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    plan = None
    requested_device_map = args.device_map
    if args.flex_percent is not None:
        try:
            policy = FlexQwenPolicy.from_sequence(args.flex_percent)
            plan = build_qwen_plan(
                args.model,
                policy,
                _compute_device(args.gpu_memory, args.flex_compute_device),
            )
        except (ValueError, OSError) as exc:
            raise SystemExit(str(exc)) from exc
        requested_device_map = plan.device_map

    config = HFOffloadConfig(
        model=args.model,
        quantization=args.quantization,
        device_map=requested_device_map,
        max_memory=_max_memory(args.gpu_memory, args.cpu_memory),
        offload_dir=args.offload_dir,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        cpu_offload=not args.no_cpu_offload,
    )
    if args.dry_run:
        output = config.to_dict()
        if plan is not None:
            output["flex_plan"] = plan.to_dict(include_parameters=args.print_flex_plan)
            if not args.print_flex_plan:
                output["flex_plan"]["stages"] = {
                    "count": len(plan.stages),
                    "hint": "add --print-flex-plan to list all stages",
                }
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0

    runner = HFOffloadLM(config, fine_grained_plan=plan).load()
    result = runner.benchmark(
        [args.prompt] * args.batch_size,
        max_new_tokens=args.gen_len,
        warmup_tokens=args.warmup_tokens,
    )
    print(result.to_json())
    if args.output_jsonl:
        append_jsonl(args.output_jsonl, result.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
