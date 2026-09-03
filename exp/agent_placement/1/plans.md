# Iteration 1 plan: remove per-tensor CUDA cache clearing

## Goal

Improve the fixed single-GPU Qwen3.8-27B decode benchmark without changing
the selected 49/51 weight placement. This isolates executor overhead from
placement quality.

Target: exceed 1 token/s. If this iteration does not meet the target, its
measurements will determine iteration 2.

## Fixed benchmark

- Physical accelerator: GPU 1 only (`CUDA_VISIBLE_DEVICES=1`)
- Model: `/models/Qwen-Qwen3.8-27B`
- Quantization: TorchAO int8 weight-only
- Logical process GPU: `cuda:0`, 15 GiB budget
- CPU budget: 35 GiB
- Placement: weights 49% GPU / 51% CPU, cache 100% GPU, activations 100% GPU
- Batch size: 1
- Prompt: `Explain tensor offloading in one sentence.`
- Warmup tokens: 1
- Generated tokens: 4
- Measurement: generated tokens divided by synchronized generation wall time
- Isolation: every measured run starts a new Python process

## Hypothesis

Accelerate 1.14 calls `set_module_tensor_to_device()` for every CPU-offloaded
tensor. Its default `clear_cache=True` clears the CUDA allocator after every
H2D materialization. Qwen has hundreds of fine-grained projection calls per
token, so repeated allocator clearing introduces synchronization and prevents
normal allocation reuse.

Keeping allocator blocks cached during generation should reduce exposed
offload latency without changing tensor homes or numerical behavior.

## Development

1. Add an explicit `fast_cpu_offload` runtime option and CLI switch.
2. For an isolated process, make Accelerate's offload hook materialization use
   `clear_cache=False`; preserve an opt-in legacy path for A/B comparison.
3. Record the runtime option in benchmark JSON.
4. Add unit tests for configuration, CLI wiring, idempotent installation, and
   output metadata.

## Test and decision rule

1. Run the legacy 49/51 command once as the control if no compatible prior
   result is available.
2. Run the fast path in a new process and append both success and failure data.
3. Confirm generated token count, GPU peak, and no OOM.
4. If throughput is greater than 1 token/s, stop after auditing the record.
5. Otherwise write `reflect.md`, identify the largest remaining exposed cost,
   and create `exp/agent_placement/2/plans.md`.
