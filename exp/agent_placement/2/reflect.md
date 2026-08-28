# Iteration 2 reflection

## Outcome

The terminal condition is satisfied. Both independent runs exceeded 1
token/s, with a median of **2.362771 token/s** and identical 9.412 GiB GPU peak
allocation. No iteration 3 is required.

Relative to the original 49/51 placement median of 0.152086 token/s, the new
median is approximately **15.54x faster**.

## Why it worked

The earlier runtime treated CPU as weight storage and repeatedly copied packed
weights through a PCIe Gen3 x4 link. Iteration 2 instead treats CPU as a compute
device:

- 31.875 GiB of logical FFN checkpoint tensors stay CPU-resident;
- FBGEMM executes gate/up/down without per-token weight transfer;
- only small hidden-state inputs and outputs cross PCIe;
- attention and DeltaNet remain GPU-resident;
- lm head no longer transfers once per generated token;
- embedding transfers only selected rows;
- unused vision tensors consume no GPU capacity.

This confirms that placement and execution location must be optimized
together. Merely changing the ordering of streamed tensors cannot overcome a
hard interconnect bandwidth bound.

## Trade-offs

- CPU FFNs use dynamic activation int8 plus per-channel int8 weights. This is
  not numerically identical to the baseline TorchAO weight-only int8 path.
- Both short deterministic runs produced the same output prefix, and the
  projection-level numerical test passed, but task-quality evaluation is
  still required before production use.
- Loading includes a one-time FFN packing phase. It is outside timed generation
  and currently takes substantially longer than the four-token benchmark.
- The reported throughput is specific to batch 1, a 9-token prompt, four
  generated tokens, this CPU, and physical GPU 1.

## Follow-up opportunities outside the terminal goal

- Serialize FBGEMM packed FFNs to remove repeat startup packing.
- Eliminate the benign Transformers warning caused by the root model reporting
  `meta` while executable submodules are correctly placed.
- Profile longer decode lengths and quality suites.
- Balance a subset of FFNs back to GPU if additional VRAM becomes available.
