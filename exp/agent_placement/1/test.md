# Iteration 1 test record

## Automated tests

Command:

```bash
/data/flexllmgen-qwen-venv/bin/python -m pytest -q \
  tests/test_hf_backend.py tests/test_qwen_flex.py \
  tests/test_qwen_flex_runtime.py
```

Result: **19 passed**, with one pre-existing TorchAO v1 configuration
deprecation warning.

`py_compile` and `git diff --check` also passed.

## Benchmark command

```bash
CUDA_VISIBLE_DEVICES=1 /data/flexllmgen-qwen-venv/bin/python \
  -m flexllmgen.hf_opt \
  --model /models/Qwen-Qwen3.8-27B \
  --quantization int8-torchao \
  --gpu-memory 15GiB --cpu-memory 35GiB \
  --flex-percent 49 51 100 0 100 0 --flex-compute-device 0 \
  --fast-cpu-offload \
  --offload-dir /data/flexllmgen-offload/agent-placement-1 \
  --local-files-only --batch-size 1 \
  --prompt 'Explain tensor offloading in one sentence.' \
  --warmup-tokens 1 --gen-len 4 \
  --output-jsonl exp/agent_placement/1/results.jsonl
```

## Results

- Status: success
- Generated tokens: 4
- Timed generation: 25.306828 seconds
- Throughput: **0.158060 token/s**
- GPU peak allocation: 12,447,214,592 bytes = 11.592 GiB
- Previous 49/51 median: 0.152086 token/s
- Improvement: 3.93%
- Target `> 1 token/s`: **not met**

The complete machine-readable result is in `results.jsonl`.

## Failed launch retained in the audit

An initial command accidentally omitted `CUDA_VISIBLE_DEVICES=1`. It addressed
physical GPU 0, which had only 1.83 GiB free, and failed during Transformers'
allocator warmup while attempting a 9.06 GiB allocation. No benchmark result
was produced. The corrected command above used physical GPU 1.
