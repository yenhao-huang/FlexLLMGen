# Iteration 1 reflection

## Conclusion

Removing per-tensor CUDA cache clearing is correct and measurable, but it is
not the dominant bottleneck. Throughput improved only 3.93%, from 0.152086 to
0.158060 token/s, while placement and GPU peak remained unchanged.

## New evidence

- Physical GPU 1 is connected at PCIe Gen3 x4, whose theoretical one-direction
  limit is about 3.94 GB/s.
- The 49/51 plan keeps 35,279,738,880 logical BF16 bytes on CPU. TorchAO's
  packed representation is roughly half that size.
- The timed decode repeatedly materializes CPU-owned gate/up, Q/QKV, norms,
  and lm-head tensors on CUDA. The required packed traffic alone establishes a
  multi-second lower bound per token on this PCIe link.
- A representative 89M-element projection took about 27.3 ms to copy as a
  TorchAO int8 tensor to GPU.
- TorchAO weight-only CPU GEMV took about 29 ms for the same projection, but
  FBGEMM dynamic-int8 execution took about 1.15 ms.

## Decision

Iteration 2 will stop streaming CPU-owned FFN weights. It will:

1. keep attention, recurrent state, norms, and lm head resident on GPU;
2. keep FFN weights resident on CPU in FBGEMM packed form;
3. execute each complete FFN on CPU and transfer only its input and final
   hidden-size output;
4. gather only referenced embedding rows on CPU;
5. keep the unused vision tower off GPU for this text-only benchmark.

This changes CPU projections from weight-only int8 to dynamic activation plus
int8 weight execution. The numerical distinction must be explicit in result
metadata and greedy output must be checked for functional validity.
