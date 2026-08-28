import json
import struct

import pytest

from flexllmgen.hf_opt import main
from flexllmgen.qwen_flex import (
    FlexQwenPolicy,
    build_qwen_plan,
)


def _write_checkpoint(tmp_path):
    tensors = {
        "model.language_model.embed_tokens.weight": 40,
        "model.language_model.layers.0.input_layernorm.weight": 4,
        "model.language_model.layers.0.self_attn.q_proj.weight": 120,
        "model.language_model.layers.0.self_attn.q_norm.weight": 4,
        "model.language_model.layers.0.self_attn.k_proj.weight": 10,
        "model.language_model.layers.0.self_attn.k_norm.weight": 4,
        "model.language_model.layers.0.self_attn.v_proj.weight": 10,
        "model.language_model.layers.0.self_attn.o_proj.weight": 60,
        "model.language_model.layers.0.post_attention_layernorm.weight": 4,
        "model.language_model.layers.0.mlp.gate_proj.weight": 80,
        "model.language_model.layers.0.mlp.up_proj.weight": 80,
        "model.language_model.layers.0.mlp.down_proj.weight": 80,
        "model.language_model.layers.1.input_layernorm.weight": 4,
        "model.language_model.layers.1.linear_attn.A_log": 2,
        "model.language_model.layers.1.linear_attn.dt_bias": 2,
        "model.language_model.layers.1.linear_attn.in_proj_qkv.weight": 100,
        "model.language_model.layers.1.linear_attn.in_proj_z.weight": 60,
        "model.language_model.layers.1.linear_attn.in_proj_b.weight": 4,
        "model.language_model.layers.1.linear_attn.in_proj_a.weight": 4,
        "model.language_model.layers.1.linear_attn.conv1d.weight": 8,
        "model.language_model.layers.1.linear_attn.norm.weight": 4,
        "model.language_model.layers.1.linear_attn.out_proj.weight": 60,
        "model.language_model.layers.1.post_attention_layernorm.weight": 4,
        "model.language_model.layers.1.mlp.gate_proj.weight": 80,
        "model.language_model.layers.1.mlp.up_proj.weight": 80,
        "model.language_model.layers.1.mlp.down_proj.weight": 80,
        "model.language_model.norm.weight": 4,
        "lm_head.weight": 40,
    }
    offset = 0
    header = {}
    for name, size in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [size],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header).encode("utf-8")
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "text_config": {
            "num_hidden_layers": 2,
            "layer_types": ["full_attention", "linear_attention"],
        },
    }))
    return tensors


def test_policy_validates_all_three_tiers():
    policy = FlexQwenPolicy.from_sequence([25, 50, 70, 30, 100, 0])
    assert policy.weight_disk_percent == 25
    assert policy.cache_disk_percent == 0
    with pytest.raises(ValueError, match="weight"):
        FlexQwenPolicy.from_sequence([60, 50, 100, 0, 100, 0])


def test_plan_splits_inside_qwen_attention_and_mlp(tmp_path):
    tensors = _write_checkpoint(tmp_path)
    plan = build_qwen_plan(
        str(tmp_path),
        FlexQwenPolicy.from_sequence([50, 50, 50, 50, 50, 50]),
        1,
    )
    by_category = {(stage.layer, stage.category): stage for stage in plan.stages}

    assert by_category[(0, "full_attention.q_proj")].home == "cpu"
    assert by_category[(0, "full_attention.v_proj")].home == 1
    assert by_category[(0, "mlp.gate_proj")].home == "cpu"
    assert by_category[(0, "mlp.down_proj")].home == 1
    assert by_category[(1, "linear_attention.in_proj_qkv")].home == "cpu"
    assert by_category[(1, "linear_attention.out_proj")].home == 1
    assert by_category[(0, "full_attention.score")].execution_device == "cpu"
    assert by_category[(1, "linear_attention.delta_rule")].execution_device == 1
    assert by_category[(0, "activation.hidden_state")].execution_device == "cpu"
    assert by_category[(1, "activation.hidden_state")].execution_device == 1
    assert sum(plan.totals_by_home().values()) == sum(tensors.values())


def test_bare_delta_parameters_remain_on_compute_device(tmp_path):
    _write_checkpoint(tmp_path)
    plan = build_qwen_plan(
        str(tmp_path),
        FlexQwenPolicy.from_sequence([0, 0, 100, 0, 100, 0]),
        1,
    )
    device_map = plan.device_map
    assert device_map["model.language_model.layers.1.linear_attn"] == 1
    assert device_map["model.language_model.layers.1.linear_attn.in_proj_qkv"] == "disk"


def test_flex_dry_run_selects_largest_gpu_budget(tmp_path, capsys):
    _write_checkpoint(tmp_path)
    assert main([
        "--model", str(tmp_path),
        "--gpu-memory", "1GiB", "15GiB",
        "--cpu-memory", "35GiB",
        "--flex-percent", "50", "50", "100", "0", "100", "0",
        "--dry-run",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["flex_plan"]["compute_device"] == 1
    assert output["flex_plan"]["stages"]["count"] > 20
