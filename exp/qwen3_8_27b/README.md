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

## Fine-grained FlexLLMGen placement

Accelerate's named strategies keep a decoder block intact.  The Qwen-specific
FlexLLMGen planner can instead place the parameter homes for each RMSNorm,
Q/K/V/O projection, DeltaNet projection/convolution/norm, and FFN
gate/up/down projection independently.  It also exposes the operations with no
checkpoint tensor as explicit runtime stages: full-attention score/softmax,
the linear-attention delta rule, and the hidden state at every decoder-block
boundary.

The six values passed to `--flex-percent` retain the original FlexLLMGen
meaning:

1. weight GPU and CPU percentages;
2. KV/recurrent-state cache GPU and CPU percentages;
3. activation GPU and CPU percentages.

The remainder of each pair is disk. Disk weight stages are supported through
per-stage streaming. Cache and activation disk homes are deliberately rejected
instead of being mislabeled; their CPU/GPU percentages must sum to 100. CPU and
disk weight homes are streamed to the primary compute GPU for their operation.
The primary GPU defaults to the entry with the largest `--gpu-memory` budget.

Inspect the complete stage graph without loading PyTorch or CUDA:

```bash
python -m flexllmgen.hf_opt \
  --model /models/Qwen-Qwen3.8-27B \
  --gpu-memory 1GiB 15GiB --cpu-memory 35GiB \
  --flex-percent 49 51 100 0 100 0 \
  --local-files-only --dry-run --print-flex-plan
```

Run the measured configuration:

```bash
python -m flexllmgen.hf_opt \
  --model /models/Qwen-Qwen3.8-27B \
  --quantization int8-torchao \
  --gpu-memory 1GiB 15GiB --cpu-memory 35GiB \
  --flex-percent 49 51 100 0 100 0 \
  --offload-dir /data/flexllmgen-offload/qwen3.8-27b-fine \
  --local-files-only --batch-size 1 --warmup-tokens 1 --gen-len 4 \
  --output-jsonl exp/qwen3_8_27b/selected_weight_results.jsonl
```

Qwen3.8 is hybrid: 48 of its 64 text blocks use Gated DeltaNet linear
attention, while every fourth block uses ordinary full attention. The planner
uses the corresponding topology for each layer rather than applying a generic
Q/K/V template to all 64 blocks.

On this host, a two-repeat search selected **49/51 GPU/CPU weights**. It
measured a median 0.152086 token/s, a worst observed 11.593 GiB peak PyTorch
allocation on GPU 1, and 3.407 GiB of headroom from the configured 15 GiB
budget. This was 10.40% faster than the 30/70 baseline. The adjacent 50/50
candidate failed both repeats during Transformers' CUDA allocator warmup, so
49/51 is the highest successful tested stage-map boundary. A mixed 75/25
GPU/CPU cache and activation diagnostic was valid but substantially slower, so
cache and activations remain 100% GPU in the recommended command.

Run or resume the recorded weight search with:

```bash
python exp/qwen3_8_27b/run_flex_weight_search.py --resume
```

The complete selection table, fixed parameters, and reproduction command are
in [`WEIGHT_SELECTION.md`](WEIGHT_SELECTION.md). Machine-readable selection,
aggregate statistics, the environment manifest, and every success/OOM record
are retained in `selected_weight_ratio.json`, `weight_search_aggregate.json`,
`weight_search_manifest.json`, and `weight_search.jsonl`, respectively.

## Heterogeneous agent placement

The 49/51 path executes all projections on GPU and streams CPU-owned weights.
Physical GPU 1 is connected through PCIe Gen3 x4, so that transfer model limits
batch-1 decode. The experimental `--agent-placement` mode instead keeps all
attention/DeltaNet/norm/lm-head tensors on GPU, packs all complete FFNs for CPU
FBGEMM execution, gathers embedding rows on CPU, and leaves the unused vision
tower off GPU.

```bash
CUDA_VISIBLE_DEVICES=1 python -m flexllmgen.hf_opt \
  --model /models/Qwen-Qwen3.8-27B \
  --quantization int8-torchao \
  --gpu-memory 15GiB --cpu-memory 35GiB \
  --agent-placement --flex-compute-device 0 --fast-cpu-offload \
  --offload-dir /data/flexllmgen-offload/agent-placement-2 \
  --local-files-only --batch-size 1 \
  --prompt 'Explain tensor offloading in one sentence.' \
  --warmup-tokens 1 --gen-len 4 \
  --output-jsonl exp/agent_placement/2/results.jsonl
```

Two independent processes measured 2.215825 and 2.509718 token/s, with a
2.362771 token/s median and identical 9.412 GiB GPU peak. CPU FFNs use dynamic
activation int8 plus per-channel int8 weights, so validate task quality before
production use. Full plan/dev/test/reflection records are in
[`exp/agent_placement`](../agent_placement/README.md).

For fair comparisons, keep `--prompt`, `--batch-size`, `--gen-len`, and
`--warmup-tokens` identical. The search runner records load or OOM failures as
rows instead of silently omitting them.
