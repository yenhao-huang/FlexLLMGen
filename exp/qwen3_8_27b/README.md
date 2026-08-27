# Qwen3.8-27B placement experiment

This experiment compares a DRAM-only 8-bit Hugging Face baseline with
FlexLLMGen's 8-bit Hugging Face-backed CPU/GPU tensor-placement candidates.
Every candidate runs in a separate process and appends its measured result to
`results.jsonl`.

The checkpoint is expected at `/models/Qwen-Qwen3.8-27B`. The model's official
safetensor index reports 55,562,855,904 bytes of BF16 tensors. Commands default
to TorchAO 8-bit weight-only quantization for accelerator candidates and
reserve 2 GiB on each GPU plus 8 GiB of available system memory. An additional
BF16 DRAM measurement in `results.jsonl` provides an unquantized reference. The
legacy `--quantization int8`
bitsandbytes path is also exposed, but on this hardware its Accelerate hooks
retained Qwen linear state until GPU OOM; `int8-torchao` is the search default.
The TorchAO path uses its interoperable affine tensor representation because
Accelerate 1.14 cannot yet materialize the newer `Int8Tensor` representation
from CPU offload. `model_logical_bytes` is the model's logical BF16 shape size,
not the packed physical size of its tensor subclasses.

Download once to the large model volume (allow at least 56 GB plus temporary
headroom), then keep benchmark runs offline:

```bash
hf download Qwen/Qwen3.8-27B \
  --local-dir /models/Qwen-Qwen3.8-27B
```

Inspect the search space without loading the model:

```bash
python exp/qwen3_8_27b/run_search.py \
  --local-files-only --dry-run
```

Run only the DRAM baseline:

```bash
python exp/qwen3_8_27b/run_search.py \
  --local-files-only --candidate dram-baseline
```

Run all generated candidates:

```bash
python exp/qwen3_8_27b/run_search.py \
  --local-files-only
```

For fair comparisons, keep `--prompt`, `--batch-size`, `--gen-len`, and
`--warmup-tokens` identical. The search runner records load or OOM failures as
rows instead of silently omitting them.
