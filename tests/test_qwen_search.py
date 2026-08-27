import importlib.util
from pathlib import Path

from flexllmgen.placement import PlacementCandidate


SCRIPT = Path(__file__).parents[1] / "exp" / "qwen3_8_27b" / "run_search.py"
SPEC = importlib.util.spec_from_file_location("qwen_search", SCRIPT)
qwen_search = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qwen_search)


def test_search_command_preserves_candidate_contract(tmp_path):
    args = qwen_search.build_parser().parse_args([
        "--model", "/models/test",
        "--offload-dir", str(tmp_path / "offload"),
        "--output-jsonl", str(tmp_path / "results.jsonl"),
        "--local-files-only",
    ])
    candidate = PlacementCandidate("balanced-85", "balanced", (18, 17), 40)

    command = qwen_search.command_for_candidate(args, candidate)

    assert command[:3] == [qwen_search.sys.executable, "-m", "flexllmgen.hf_opt"]
    assert command[command.index("--gpu-memory") + 1:] == [
        "18GiB", "17GiB", "--local-files-only"
    ]
    assert command[command.index("--cpu-memory") + 1] == "40GiB"
    assert command[command.index("--quantization") + 1] == "int8-torchao"


def test_cpu_search_command_does_not_pass_gpu_memory(tmp_path):
    args = qwen_search.build_parser().parse_args([
        "--output-jsonl", str(tmp_path / "results.jsonl")
    ])
    candidate = PlacementCandidate("dram-baseline", "cpu", (0, 0), 40)

    command = qwen_search.command_for_candidate(args, candidate)

    assert "--gpu-memory" not in command
    assert command[command.index("--quantization") + 1] == "int8-torchao"
    assert command[command.index("--dtype") + 1] == "auto"


def test_bitsandbytes_cpu_baseline_is_labeled_bfloat16(tmp_path):
    args = qwen_search.build_parser().parse_args([
        "--quantization", "int8",
        "--output-jsonl", str(tmp_path / "results.jsonl"),
    ])
    candidate = PlacementCandidate("dram-baseline", "cpu", (0, 0), 40)

    command = qwen_search.command_for_candidate(args, candidate)

    assert command[command.index("--quantization") + 1] == "none"
    assert command[command.index("--dtype") + 1] == "bfloat16"
