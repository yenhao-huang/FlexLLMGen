import pytest


torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="fine-grained Qwen runtime integration needs CUDA device 1",
)


def test_agent_fbgemm_conversion_matches_torchao_projection():
    pytest.importorskip("torchao")
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    from flexllmgen.qwen_flex import _aqt_to_fbgemm_dynamic_linear

    torch.manual_seed(0)
    reference = torch.nn.Linear(32, 64, bias=False).eval()
    quantize_(
        reference,
        Int8WeightOnlyConfig(group_size=None, version=1),
    )
    converted = _aqt_to_fbgemm_dynamic_linear(reference)
    inputs = torch.randn(3, 32)

    with torch.inference_mode():
        expected = reference(inputs)
        actual = converted(inputs)

    assert type(converted).__module__.startswith("torch.ao.nn.quantized.dynamic")
    torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.08)


def test_tiny_qwen_mixes_disk_weights_cpu_cores_and_cpu_activations(tmp_path):
    pytest.importorskip("torchao")
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import TorchAoConfig
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    from flexllmgen.qwen_flex import (
        FlexQwenPolicy,
        build_qwen_plan,
        install_qwen_fine_grained_runtime,
    )

    text = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=32,
        num_position_embeddings=16,
    )
    config = Qwen3_5Config(
        text_config=text,
        vision_config=vision,
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    Qwen3_5ForConditionalGeneration(config).eval().save_pretrained(tmp_path)

    plan = build_qwen_plan(
        str(tmp_path),
        FlexQwenPolicy.from_sequence([25, 25, 50, 50, 50, 50]),
        1,
    )
    offload_dir = tmp_path / "offload"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        tmp_path,
        device_map=plan.quantization_load_device_map,
        quantization_config=TorchAoConfig(
            quant_type=Int8WeightOnlyConfig(group_size=None, version=1)
        ),
        offload_folder=offload_dir,
        dtype="auto",
    ).eval()
    install_qwen_fine_grained_runtime(model, plan, str(offload_dir))

    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([[1, 3, 4]], device="cuda:1"),
            use_cache=True,
        )

    assert output.logits.shape == (1, 3, 64)
    assert output.logits.device == torch.device("cuda:1")
    assert list((offload_dir / "fine-grained").glob("*.pt"))
