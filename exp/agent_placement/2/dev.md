# Iteration 2 development record

## Agent placement strategy

Added `--agent-placement`, mutually exclusive with `--flex-percent`. The plan
is named `agent_cpu_ffn_v1` in dry-run and benchmark metadata.

Logical checkpoint placement:

- GPU: 17,023,679,488 bytes = 15.855 GiB
- CPU: 37,689,777,632 bytes = 35.101 GiB
- Disk: 0 bytes

These are BF16 checkpoint logical bytes. TorchAO/FBGEMM packed int8 physical
storage is smaller.

## Runtime implementation

- All attention, DeltaNet, normalization, recurrent/cache state, final norm,
  and lm-head stages are mapped to the compute GPU.
- All gate/up/down projections are initially loaded through CPU-offload hooks.
- After loading, each TorchAO per-row affine-int8 FFN weight is converted
  directly from its integer data/scales/zero-points into a per-channel qint8
  tensor and packed into a CPU FBGEMM dynamic-linear module.
- Each complete MLP is replaced atomically. Its GPU input is copied to CPU as
  float32, gate/up/activation/down execute on CPU, and only the final
  hidden-size result returns to the original GPU dtype/device.
- The embedding hook is replaced with CPU row gather. TorchAO does not quantize
  `nn.Embedding`, so this checkpoint keeps the BF16 table in DRAM and returns
  only rows referenced by the requested token IDs.
- Vision weights remain on CPU and are not invoked for a text-only prompt.

## Numerical behavior

CPU FFNs use dynamic activation quantization plus per-channel int8 weights,
whereas the baseline uses float activations plus weight-only int8. This is a
deliberate numerical change and is recorded through the strategy name. A unit
comparison against the TorchAO reference projection passed with 8% relative
and 0.08 absolute tolerance.

## Files

- `flexllmgen/qwen_flex.py`: agent planner, TorchAO-to-FBGEMM conversion,
  heterogeneous MLP, and sparse dense/quantized embedding runtime.
- `flexllmgen/hf_opt.py`: `--agent-placement` CLI option and validation.
- `tests/test_qwen_flex.py`: plan/device/CLI tests.
- `tests/test_qwen_flex_runtime.py`: projection conversion numerical test.
