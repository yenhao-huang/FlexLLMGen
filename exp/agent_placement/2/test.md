# Iteration 2 test record

## Automated tests

Full command:

```bash
/data/flexllmgen-qwen-venv/bin/python -m pytest -q tests
python3 -m py_compile \
  flexllmgen/hf_backend.py flexllmgen/hf_opt.py \
  flexllmgen/qwen_flex.py exp/qwen3_8_27b/run_flex_weight_search.py
git diff --check
```

Result: **33 passed**, with two TorchAO v1 configuration deprecation
warnings. Python compilation and whitespace validation passed.

## Benchmark command

Both successful measurements ran this command in a new process:

```bash
CUDA_VISIBLE_DEVICES=1 /data/flexllmgen-qwen-venv/bin/python \
  -m flexllmgen.hf_opt \
  --model /models/Qwen-Qwen3.8-27B \
  --quantization int8-torchao \
  --gpu-memory 15GiB --cpu-memory 35GiB \
  --agent-placement --flex-compute-device 0 \
  --fast-cpu-offload \
  --offload-dir /data/flexllmgen-offload/agent-placement-2 \
  --local-files-only --batch-size 1 \
  --prompt 'Explain tensor offloading in one sentence.' \
  --warmup-tokens 1 --gen-len 4 \
  --output-jsonl exp/agent_placement/2/results.jsonl
```

## Successful results

| Repeat | Generated | Elapsed seconds | Tokens/s | GPU peak GiB | Output prefix |
|---:|---:|---:|---:|---:|:---|
| 1 | 4 | 1.593805 | 2.509718 | 9.412 | `Tensor offloading` |
| 2 | 4 | 1.805197 | 2.215825 | 9.412 | `Tensor offloading` |

- Successes: 2/2
- Median: **2.362771 token/s**
- Range: 2.215825–2.509718 token/s
- Original 49/51 median: 0.152086 token/s
- Median speedup: 15.54x
- Target `> 1 token/s`: **met by both repeats**

Machine-readable rows, including complete device maps, exact logical-byte
placement, generated text, and GPU peaks, are in `results.jsonl`.

## Failed development run

The first integration attempt completed all FFN packing but rejected the
embedding because the implementation assumed TorchAO had quantized
`nn.Embedding`. The checkpoint embedding is BF16. Sparse row gather was
extended to support dense embedding tables, tests were rerun, and both
subsequent benchmark processes succeeded. No failed row was appended to
`results.jsonl` because generation never began.
