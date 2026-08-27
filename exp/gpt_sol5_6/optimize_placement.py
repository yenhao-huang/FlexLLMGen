"""Ask GPT-5.6 Sol for bounded Qwen tensor-placement candidates.

The model proposes candidates; local validation remains authoritative. No
candidate is benchmarked by this script, so its output can be reviewed before
passing the selected plans to ``flexllmgen.hf_opt``.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Mapping, Sequence


STRATEGIES = {"auto", "balanced", "balanced_low_0", "sequential"}

OUTPUT_SCHEMA = {
    "type": "json_schema",
    "name": "tensor_placement_proposal",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "analysis_summary": {"type": "string"},
            "candidates": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "strategy": {
                            "type": "string",
                            "enum": sorted(STRATEGIES),
                        },
                        "gpu_memory_gib": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 1},
                        },
                        "cpu_memory_gib": {"type": "integer", "minimum": 1},
                        "batch_size": {"type": "integer", "minimum": 1},
                        "hypothesis": {"type": "string"},
                    },
                    "required": [
                        "name",
                        "strategy",
                        "gpu_memory_gib",
                        "cpu_memory_gib",
                        "batch_size",
                        "hypothesis",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["analysis_summary", "candidates"],
        "additionalProperties": False,
    },
}


def validate_proposal(
    proposal: Mapping[str, Any],
    gpu_free_gib: Sequence[int],
    cpu_available_gib: int,
    gpu_reserve_gib: int = 2,
    cpu_reserve_gib: int = 8,
) -> Dict[str, Any]:
    """Validate the untrusted API response against local hardware limits."""
    if not isinstance(proposal, Mapping):
        raise ValueError("proposal must be an object")
    if not isinstance(proposal.get("analysis_summary"), str):
        raise ValueError("proposal.analysis_summary must be a string")
    candidates = proposal.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 5:
        raise ValueError("proposal must contain one to five candidates")

    gpu_limits = [max(1, value - gpu_reserve_gib) for value in gpu_free_gib]
    cpu_limit = max(1, cpu_available_gib - cpu_reserve_gib)
    validated: List[Dict[str, Any]] = []
    names = set()
    required = {
        "name", "strategy", "gpu_memory_gib", "cpu_memory_gib",
        "batch_size", "hypothesis",
    }
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping) or set(candidate) != required:
            raise ValueError("candidate {} has unexpected fields".format(index))
        name = candidate["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("candidate names must be non-empty and unique")
        names.add(name)
        if candidate["strategy"] not in STRATEGIES:
            raise ValueError("candidate {} has an invalid strategy".format(name))
        memory = candidate["gpu_memory_gib"]
        if not isinstance(memory, list) or len(memory) != len(gpu_limits):
            raise ValueError("candidate {} must specify every GPU".format(name))
        if any(
            not isinstance(value, int) or value < 1 or value > gpu_limits[i]
            for i, value in enumerate(memory)
        ):
            raise ValueError("candidate {} exceeds a GPU memory limit".format(name))
        cpu_memory = candidate["cpu_memory_gib"]
        if not isinstance(cpu_memory, int) or not 1 <= cpu_memory <= cpu_limit:
            raise ValueError("candidate {} exceeds the CPU memory limit".format(name))
        batch_size = candidate["batch_size"]
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("candidate {} has an invalid batch size".format(name))
        if not isinstance(candidate["hypothesis"], str) or not candidate["hypothesis"]:
            raise ValueError("candidate {} needs a hypothesis".format(name))
        validated.append(dict(candidate))
    return {
        "analysis_summary": proposal["analysis_summary"],
        "candidates": validated,
    }


def request_proposal(args, hardware: Mapping[str, Any], prior_results: Sequence[Mapping[str, Any]]):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Install the GPT optimizer dependency with `pip install -e '.[sol]'`."
        ) from exc
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to run the GPT-5.6 Sol optimizer")

    client = OpenAI()
    payload = {
        "objective": "Maximize measured end-to-end generated tokens per second.",
        "model": "Qwen/Qwen3.8-27B",
        "model_bf16_weight_bytes": 55_562_855_904,
        "quantization": {
            "candidate_backend": "TorchAO int8 weight-only",
            "compatibility_format": "v1 affine tensor, per-row scales",
            "failed_baseline": "bitsandbytes int8 CPU offload exceeded its GPU budget",
        },
        "architecture": {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "pattern": "3 linear-attention layers then 1 full-attention layer",
        },
        "hardware": hardware,
        "constraints": {
            "gpu_reserve_gib": args.gpu_reserve_gib,
            "cpu_reserve_gib": args.cpu_reserve_gib,
            "allowed_strategies": sorted(STRATEGIES),
            "maximum_candidates": 5,
        },
        "prior_benchmark_results": list(prior_results),
    }
    response = client.responses.create(
        model=args.model,
        reasoning={"effort": args.reasoning_effort},
        instructions=(
            "You optimize tensor placement for reproducible LLM inference benchmarks. "
            "Respect every stated memory reserve. Propose diverse, testable candidates; "
            "do not claim a candidate is best before it is measured."
        ),
        input=json.dumps(payload, sort_keys=True),
        text={"format": OUTPUT_SCHEMA, "verbosity": "low"},
        max_output_tokens=args.max_output_tokens,
        store=False,
    )
    proposal = json.loads(response.output_text)
    validated = validate_proposal(
        proposal,
        hardware["gpu_free_gib"],
        hardware["cpu_available_gib"],
        args.gpu_reserve_gib,
        args.cpu_reserve_gib,
    )
    validated["response_id"] = response.id
    validated["model"] = response.model
    usage = getattr(response, "usage", None)
    validated["usage"] = usage.model_dump() if usage is not None else None
    return validated


def _read_jsonl(path: str) -> List[Mapping[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="high",
    )
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--gpu-free-gib", type=int, nargs="+", required=True)
    parser.add_argument("--cpu-available-gib", type=int, required=True)
    parser.add_argument("--gpu-reserve-gib", type=int, default=2)
    parser.add_argument("--cpu-reserve-gib", type=int, default=8)
    parser.add_argument("--prior-results", default="exp/qwen3_8_27b/results.jsonl")
    parser.add_argument("--output", default="exp/gpt_sol5_6/proposal.json")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    hardware = {
        "gpu_free_gib": args.gpu_free_gib,
        "cpu_available_gib": args.cpu_available_gib,
    }
    proposal = request_proposal(args, hardware, _read_jsonl(args.prior_results))
    output = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as destination:
        json.dump(proposal, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(json.dumps(proposal, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
