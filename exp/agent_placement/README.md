# Agent placement experiment

The experiment reached the `> 1 token/s` target in iteration 2 on physical GPU
1 with batch size 1.

| Iteration | Strategy | Median/observed tokens/s | Outcome |
|---:|:---|---:|:---|
| 1 | 49/51 streaming, allocator-cache fast path | 0.158060 | continue |
| 2 | GPU attention + CPU FBGEMM FFN | 2.362771 median | target met |

Iteration records:

- `1/plans.md`, `1/dev.md`, `1/test.md`, `1/reflect.md`
- `2/plans.md`, `2/dev.md`, `2/test.md`, `2/reflect.md`

Use the exact successful command documented in `2/test.md`. Raw successful
measurements are stored in `2/results.jsonl`.
