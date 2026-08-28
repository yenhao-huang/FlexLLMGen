# Iteration 1 development record

## Change

Added the explicit `--fast-cpu-offload` runtime option.

When enabled, the process-local Accelerate offload hook calls
`set_module_tensor_to_device(..., clear_cache=False)`. The hook still replaces
each temporary CUDA weight with a meta tensor after its module returns; only
the freed CUDA allocator block remains cached for reuse by subsequent stages.

The option is deliberately explicit so legacy and optimized runs can be
measured in independent processes. Benchmark JSON includes
`fast_cpu_offload` to prevent mixing the two modes.

## Files

- `flexllmgen/hf_backend.py`: idempotent Accelerate fast-path installation,
  configuration field, and result metadata.
- `flexllmgen/hf_opt.py`: `--fast-cpu-offload` CLI switch.
- `tests/test_hf_backend.py`: dry-run wiring and cache-clear behavior tests.

## Invariants

- Placement remains `49 51 100 0 100 0`.
- CPU tensors remain owned by Accelerate's CPU weights map.
- Computation remains on the single visible CUDA device.
- No numerical kernels or sampling behavior are changed.
- Legacy behavior remains available by omitting `--fast-cpu-offload`.
