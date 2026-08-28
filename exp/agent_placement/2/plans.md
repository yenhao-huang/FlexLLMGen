# Iteration 2 plan: heterogeneous CPU FFN execution

## Goal

Remove repeated FFN weight traffic over GPU 1's PCIe Gen3 x4 link. Continue to
use a single physical GPU, batch size 1, and the same prompt/warmup/generation
length. Target remains greater than 1 token/s.

## Placement

- GPU persistent: all full-attention projections, all DeltaNet projections and
  recurrent state, all normalization/control tensors, final norm, and lm head.
- CPU persistent: all 64 FFNs in FBGEMM packed int8 form.
- CPU sparse access: token embedding table; gather/dequantize only requested
  rows and return them to GPU.
- CPU inactive: vision tower for the text-only request.
- Cache and hidden-state boundaries: GPU.

For each FFN, gate/up/down remain co-located and execute as one CPU subgraph:

```text
GPU hidden -> CPU float activation -> FBGEMM gate/up -> activation
           -> FBGEMM down -> GPU model dtype
```

Only hidden-size activation tensors cross PCIe; weights never cross during
generation.

## Development

1. Add an explicit agent placement strategy and machine-readable strategy name.
2. Convert TorchAO CPU affine-int8 projection weights into per-channel qint8
   tensors and FBGEMM dynamic-linear packed modules after checkpoint loading.
3. Replace each decoder FFN atomically, avoiding mixed-device intermediate
   tensors and releasing obsolete Accelerate hooks.
4. Add a quantized embedding row-gather module.
5. Record the heterogeneous execution mode in benchmark JSON.
6. Add numerical unit tests against TorchAO reference projections and a tiny
   Qwen integration test.

## Test and decision rule

Run the same isolated benchmark as iteration 1 with the agent strategy. Record
success/OOM, generated token count, decoded output, elapsed time, GPU peak, and
throughput. If throughput does not exceed 1 token/s, profile the remaining GPU
attention, CPU FFN, embedding, and lm-head segments separately before defining
iteration 3.
