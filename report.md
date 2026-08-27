# Qwen3.8-27B 8-bit offload development report

Date: 2026-08-27

## Outcome

FlexLLMGen now has an additive Hugging Face backend for Qwen3.8 and other
modern architectures. It supports validated CPU/GPU placement, TorchAO
weight-only int8 inference, a legacy bitsandbytes compatibility mode,
reproducible JSONL benchmarks, a bounded placement search, and a GPT-5.6 Sol
candidate generator.

The best comparable measurement was the TorchAO int8 `sequential` placement
with requested memory budgets of 1 GiB on GPU 0, 15 GiB on GPU 1, and 35 GiB
on CPU. It achieved **0.18938 generated token/s**, or **1.98x** the matching
DRAM-only int8 baseline of **0.09550 token/s**.

## Checkpoint and environment

- Model: `Qwen/Qwen3.8-27B`, downloaded to `/models/Qwen-Qwen3.8-27B`
- Revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Verification: all 18 safetensor shards and all 1,199 index entries present;
  indexed tensor size is 55,562,855,904 bytes
- Architecture: `Qwen3_5ForConditionalGeneration`, 64 text layers, hidden size
  5,120, with a repeating three-linear-attention/one-full-attention pattern
- Hardware: 2 x NVIDIA GeForce RTX 4090 (24,564 MiB each), 61 GiB RAM, 121 GiB
  swap
- Software used for the successful int8 runs: Python 3.12, PyTorch
  2.13.0+cu130, Transformers 5.15.1, Accelerate 1.14.0, TorchAO 0.17.0

The checkpoint follows the official
[Qwen model repository](https://huggingface.co/Qwen/Qwen3.8-27B/tree/main) and
[configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json).

## Method

The comparable rows use the same nine-token prompt, batch size 1, one generated
warmup token, four measured generated tokens, deterministic greedy generation,
and TorchAO int8 weight-only quantization. The timer includes prompt processing
and measured generation but excludes model loading and warmup. Each candidate
runs in a fresh process.

GPU availability was asymmetric because unrelated workloads occupied the
machine: approximately 2.2 GiB was free on GPU 0 and 18.0 GiB on GPU 1. The
search therefore kept GPU 0 at a 1 GiB requested budget and explored larger
budgets on GPU 1. Peak memory is PyTorch allocated memory during the timed
generation, not total board usage.

## Results

| Backend and placement | Workload | Throughput | GPU peak allocation | Result |
| --- | --- | ---: | ---: | --- |
| BF16, all weights in DRAM | 1 token, no warmup | 0.00546 tok/s | 0 | Reference only |
| bitsandbytes int8, sequential 1/15 GiB | 1 token, no warmup | - | 17.49 GiB allocated on GPU 1 | OOM |
| bitsandbytes int8, sequential 1/11 GiB | 1 token, no warmup | - | 17.49 GiB allocated on GPU 1 | OOM |
| TorchAO int8, sequential 1/11 GiB | 1 token, no warmup | 0.01976 tok/s | 12.44 GiB on GPU 1 | Pass |
| TorchAO int8, sequential 1/15 GiB | 1 token, no warmup | 0.06001 tok/s | 16.00 GiB on GPU 1 | Pass |
| TorchAO int8, balanced 1/15 GiB | 1 token, no warmup | 0.09146 tok/s | 16.00 GiB on GPU 1 | Pass |
| TorchAO int8, all weights in DRAM | 4 tokens, 1 warmup | 0.09550 tok/s | 0 | Comparable baseline |
| **TorchAO int8, sequential 1/15 GiB** | **4 tokens, 1 warmup** | **0.18938 tok/s** | **16.00 GiB on GPU 1** | **Best comparable: 1.98x** |

The backend verified 606 TorchAO quantized parameter tensors in both comparable
int8 runs. `model_logical_bytes` remains 54,713,457,448 because PyTorch tensor
subclasses expose their logical BF16 shape and dtype; it must not be interpreted
as packed physical storage. The canonical machine-readable rows are in
`exp/qwen3_8_27b/results.jsonl`.

The older bitsandbytes path exceeded the requested GPU budget during its first
forward pass at both tested budgets. TorchAO's newer v2 `Int8Tensor` format
currently conflicts with Accelerate 1.14's CPU-offload dtype conversion, so the
backend explicitly selects TorchAO's interoperable affine int8 representation
with per-row scales. This is isolated behind the `int8-torchao` option and can
be changed when the upstream hook supports v2 tensors.

## Reproduce the best result

```bash
pip install -e '.[qwen]'
python -m flexllmgen.hf_opt \
  --model /models/Qwen-Qwen3.8-27B \
  --quantization int8-torchao --device-map sequential \
  --gpu-memory 1GiB 15GiB --cpu-memory 35GiB \
  --local-files-only --batch-size 1 \
  --prompt 'Explain tensor offloading in one sentence.' \
  --warmup-tokens 1 --gen-len 4
```

Run `python exp/qwen3_8_27b/run_search.py --local-files-only --dry-run` to
derive a search space from current free memory before running candidates.

## GPT-5.6 Sol optimizer

`exp/gpt_sol5_6/optimize_placement.py` makes one bounded Responses API request
to `gpt-5.6-sol`, requests strict structured output, caps the proposal at five
candidates, validates every memory budget locally, stores no API-side response
state, and never executes a proposal automatically. The implementation follows
the official [GPT-5.6 Sol model contract](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
and [latest-model guidance](https://developers.openai.com/api/docs/guides/latest-model).

The live Sol call was not made because this environment has neither the OpenAI
SDK nor `OPENAI_API_KEY`. No proposal or benchmark result was fabricated. Once
credentials are supplied, install `.[sol]`, run the documented command, review
the locally validated proposal, and benchmark it with the same workload above.

## GitHub tracking

The canonical upstream repository is archived, and the active fork initially
had Issues disabled. Issues were enabled on the fork so the requested work is
tracked there:

- [Issue #1: Qwen3.8-27B 8-bit support](https://github.com/yenhao-huang/FlexLLMGen/issues/1)
- [Issue #2: reproducible tensor-placement search](https://github.com/yenhao-huang/FlexLLMGen/issues/2)
- [Issue #3: GPT-5.6 Sol placement workflow](https://github.com/yenhao-huang/FlexLLMGen/issues/3)

## Validation and limitations

- Unit tests: 18 passed
- Patch hygiene: `git diff --check` passed
- Qwen smoke test: the 0.8B checkpoint completed mixed CPU/GPU generation
- Full model: both DRAM-only and mixed-placement 27B int8 generation completed
- Disk: the model consumed about 55.6 GB; no duplicate checkpoint was created

These are small batch-1 engineering measurements on a shared machine, not a
production throughput study. The single-token rows show substantial run-to-run
variance and are included as search evidence, while the matched four-token rows
are the comparison used for the reported speedup. The new backend uses
Transformers/Accelerate placement and does not retrofit Qwen into FlexLLMGen's
legacy OPT-specific block scheduler.
