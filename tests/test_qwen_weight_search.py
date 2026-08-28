import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "exp" / "qwen3_8_27b" / "run_flex_weight_search.py"
SPEC = importlib.util.spec_from_file_location("qwen_weight_search", SCRIPT)
search = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(search)


def _success(candidate, repeat, throughput, peak_gib=10):
    return {
        "candidate_weight_gpu_percent": candidate,
        "repeat": repeat,
        "status": "success",
        "tokens_per_second": throughput,
        "elapsed_seconds": 4 / throughput,
        "compute_device": 1,
        "gpu_peak_memory_bytes": {"1": int(peak_gib * search.GIB)},
    }


def test_selection_uses_median_throughput_and_requires_headroom():
    rows = [
        _success(30, 1, 0.13, 10),
        _success(30, 2, 0.15, 10.5),
        _success(45, 1, 0.20, 14.2),
        _success(45, 2, 0.21, 14.2),
        _success(50, 1, 0.30, 12),
        {
            "candidate_weight_gpu_percent": 50,
            "repeat": 2,
            "status": "failed",
        },
    ]
    aggregates = search.aggregate_rows(
        rows,
        candidates=[30, 45, 50],
        repeats=2,
        gpu_budget_gib=15,
        required_headroom_gib=1,
    )
    selected = search.select_best(aggregates)

    assert selected["weight_gpu_percent"] == 30
    assert selected["median_tokens_per_second"] == 0.14
    assert aggregates[1]["stable"] is True
    assert aggregates[1]["eligible"] is False
    assert aggregates[2]["stable"] is False


def test_selection_tie_breaks_on_lower_peak_then_percentage():
    rows = [
        _success(30, 1, 0.1, 11),
        _success(40, 1, 0.1, 10),
        _success(45, 1, 0.1, 10),
    ]
    aggregates = search.aggregate_rows(rows, [30, 40, 45], 1, 15, 1)
    assert search.select_best(aggregates)["weight_gpu_percent"] == 40


def test_child_command_fixes_cache_activation_and_benchmark(tmp_path):
    args = search.build_parser().parse_args([
        "--model", "/models/test",
        "--result-dir", str(tmp_path),
    ])
    command = search._command(args, 33, tmp_path / "child.jsonl")
    flex_index = command.index("--flex-percent")

    assert command[flex_index + 1:flex_index + 7] == ["33", "67", "100", "0", "100", "0"]
    assert command[command.index("--warmup-tokens") + 1] == "1"
    assert command[command.index("--gen-len") + 1] == "4"
    assert command[command.index("--batch-size") + 1] == "1"


def test_failure_reason_classifies_cuda_oom():
    assert search._failure_reason({
        "error_tail": "torch.OutOfMemoryError: CUDA out of memory",
        "returncode": 1,
    }) == "cuda_oom"
