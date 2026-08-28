import json
import sys
import types

import pytest

from flexllmgen.hf_backend import HFOffloadConfig, _install_accelerate_fast_cpu_offload
from flexllmgen.hf_opt import main
from flexllmgen.placement import GIB, PlacementCandidate, generate_candidates


class FakeBitsAndBytesConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTransformers:
    BitsAndBytesConfig = FakeBitsAndBytesConfig

    class TorchAoConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs


def test_int8_model_kwargs_enable_cpu_offload(tmp_path):
    config = HFOffloadConfig(
        model="Qwen/Qwen3.8-27B",
        quantization="int8",
        device_map="balanced",
        max_memory={0: "20GiB", "cpu": "32GiB"},
        offload_dir=str(tmp_path),
    )

    kwargs = config.model_kwargs(FakeTransformers)

    assert kwargs["device_map"] == "balanced"
    assert kwargs["max_memory"] == {0: "20GiB", "cpu": "32GiB"}
    assert kwargs["quantization_config"].kwargs == {
        "load_in_8bit": True,
        "llm_int8_enable_fp32_cpu_offload": True,
    }


def test_cpu_baseline_uses_explicit_root_device_map():
    config = HFOffloadConfig(
        model="Qwen/Qwen3.8-27B",
        quantization="none",
        device_map="cpu",
        max_memory={"cpu": "48GiB"},
    )

    kwargs = config.model_kwargs(FakeTransformers)

    assert kwargs["device_map"] == {"": "cpu"}
    assert "max_memory" not in kwargs
    assert "offload_folder" not in kwargs


def test_int8_cpu_only_configuration_is_rejected():
    with pytest.raises(ValueError, match="requires an accelerator"):
        HFOffloadConfig(model="Qwen/Qwen3.8-27B", device_map="cpu")


def test_torchao_int8_allows_cpu_only_configuration():
    config = HFOffloadConfig(
        model="Qwen/Qwen3.8-27B",
        quantization="int8-torchao",
        device_map="cpu",
    )
    assert config.resolved_device_map == {"": "cpu"}


def test_torchao_uses_accelerate_compatible_per_row_format(monkeypatch):
    class FakeInt8WeightOnlyConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    torchao = types.ModuleType("torchao")
    quantization = types.ModuleType("torchao.quantization")
    quantization.Int8WeightOnlyConfig = FakeInt8WeightOnlyConfig
    torchao.quantization = quantization
    monkeypatch.setitem(sys.modules, "torchao", torchao)
    monkeypatch.setitem(sys.modules, "torchao.quantization", quantization)

    config = HFOffloadConfig(
        model="Qwen/Qwen3.8-27B",
        quantization="int8-torchao",
        device_map="cpu",
    )

    kwargs = config.model_kwargs(FakeTransformers)

    quant_type = kwargs["quantization_config"].kwargs["quant_type"]
    assert quant_type.kwargs == {"group_size": None, "version": 1}


@pytest.mark.parametrize(
    "field,value",
    [
        ("quantization", "int4"),
        ("device_map", "magic"),
        ("dtype", "int8"),
    ],
)
def test_invalid_options_fail_at_config_boundary(field, value):
    kwargs = {"model": "Qwen/Qwen3.8-27B", field: value}
    with pytest.raises(ValueError):
        HFOffloadConfig(**kwargs)


def test_memory_values_require_units():
    with pytest.raises(ValueError, match="byte unit"):
        HFOffloadConfig(model="model", max_memory={0: "20"})


def test_candidate_estimates_int8_weight_tiers():
    candidate = PlacementCandidate("test", "balanced", (10, 10), 4)

    placement = candidate.estimated_weight_placement(64 * GIB, bits=8)

    assert placement == {"gpu": 20 * GIB, "cpu": 4 * GIB, "disk": 8 * GIB}


def test_candidate_search_is_ordered_and_reserves_memory():
    candidates = generate_candidates((24, 24), 60)

    assert candidates[0].name == "dram-baseline"
    assert candidates[0].gpu_memory_gib == (0, 0)
    assert candidates[1].gpu_memory_gib == (11, 11)
    assert candidates[-1].gpu_memory_gib == (20, 20)
    assert all(candidate.cpu_memory_gib == 52 for candidate in candidates)


def test_dry_run_prints_stable_json(capsys):
    assert main([
        "--model", "Qwen/Qwen3.8-27B",
        "--device-map", "balanced",
        "--gpu-memory", "20GiB", "20GiB",
        "--cpu-memory", "32GiB",
        "--dry-run",
    ]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["model"] == "Qwen/Qwen3.8-27B"
    assert output["max_memory"] == {"0": "20GiB", "1": "20GiB", "cpu": "32GiB"}


def test_fast_cpu_offload_is_exposed_in_dry_run(capsys):
    assert main([
        "--model", "Qwen/Qwen3.8-27B",
        "--device-map", "balanced",
        "--fast-cpu-offload",
        "--dry-run",
    ]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["fast_cpu_offload"] is True


def test_fast_cpu_offload_disables_default_cache_clear(monkeypatch):
    import accelerate.hooks as accelerate_hooks

    calls = []

    def fake_set_module_tensor_to_device(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        accelerate_hooks,
        "set_module_tensor_to_device",
        fake_set_module_tensor_to_device,
    )

    assert _install_accelerate_fast_cpu_offload() is True
    assert _install_accelerate_fast_cpu_offload() is False
    accelerate_hooks.set_module_tensor_to_device("module", "weight", "cuda:0")

    assert calls[0][1]["clear_cache"] is False
