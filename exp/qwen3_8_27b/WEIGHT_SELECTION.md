# Qwen3.8-27B fine-grained weight percentage selection

## Fixed benchmark parameters

- Model: `/models/Qwen-Qwen3.8-27B`
- Weight GPU candidates: `15, 20, 30, 33, 40, 45, 49, 50`
- Repeats per candidate: 2
- Isolation: every repeat ran in a new Python process
- Python: `/data/flexllmgen-qwen-venv/bin/python`

```json
{
  "activation_cpu_percent": 0,
  "activation_gpu_percent": 100,
  "batch_size": 1,
  "cache_cpu_percent": 0,
  "cache_gpu_percent": 100,
  "compute_device": 1,
  "cpu_memory": "35GiB",
  "gen_len": 4,
  "gpu_memory": [
    "1GiB",
    "15GiB"
  ],
  "local_files_only": true,
  "prompt": "Explain tensor offloading in one sentence.",
  "quantization": "int8-torchao",
  "warmup_tokens": 1
}
```

## Results

| GPU/CPU weight % | Success | Median tok/s | Range tok/s | Peak GPU GiB | Budget headroom GiB | Eligible |
|---:|---:|---:|---:|---:|---:|:---:|
| 15/85 | 2/2 | 0.104047 | 0.103857–0.104237 | 4.411 | 10.589 | yes |
| 20/80 | 2/2 | 0.137171 | 0.137163–0.137180 | 9.858 | 5.142 | yes |
| 30/70 | 2/2 | 0.137756 | 0.137637–0.137874 | 9.885 | 5.115 | yes |
| 33/67 | 2/2 | 0.137081 | 0.137035–0.137126 | 9.981 | 5.019 | yes |
| 40/60 | 2/2 | 0.145673 | 0.145658–0.145688 | 10.056 | 4.944 | yes |
| 45/55 | 2/2 | 0.151365 | 0.151356–0.151374 | 11.467 | 3.533 | yes |
| 49/51 | 2/2 | 0.152086 | 0.152085–0.152088 | 11.593 | 3.407 | yes |
| 50/50 | 0/2 | - | - | - | - | no |

## Selected ratio

Selected **49% GPU / 51% CPU** for weights.

Selection rule: highest median measured throughput among candidates that completed all repeats and retained at least 1.0 GiB below the 15.0 GiB PyTorch allocation budget.

- Median throughput: 0.152086 token/s
- Worst observed GPU peak: 11.593 GiB
- Remaining budget headroom: 3.407 GiB
- Checkpoint logical bytes by home: `{"1": 19433718240, "cpu": 35279738880}`
- Throughput improvement over 30/70: 10.40%
- Throughput improvement over 45/55: 0.48%

### Reproduction command

```bash
source /data/flexllmgen-qwen-venv/bin/activate
cd /data/FlexLLMGen
python -m flexllmgen.hf_opt --model /models/Qwen-Qwen3.8-27B --quantization int8-torchao --gpu-memory 1GiB 15GiB --cpu-memory 35GiB --flex-percent 49 51 100 0 100 0 --flex-compute-device 1 --offload-dir /data/flexllmgen-offload/qwen3.8-27b-fine --local-files-only --batch-size 1 --warmup-tokens 1 --gen-len 4 --prompt 'Explain tensor offloading in one sentence.' --output-jsonl exp/qwen3_8_27b/selected_weight_results.jsonl
```

## Failures

- 50/50 failed 2/2 repeats: `{"cuda_oom": 2}`.

Raw success and failure records are in `weight_search.jsonl`; per-run stdout/stderr is in `weight_search_runs/`.
