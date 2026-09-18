# GPU prepared-execution overhead investigation

Date: 2026-09-18

Investigation of why tenferro CUDA loses to PyTorch on the A100 GPU benchmark
suite, and of the per-op overhead that dominates small operations. Measurements
were taken with the `nvidia-gpu` suites in `tensor4all/tenferro-benchmark`
(PR #102 raw runs, tenferro-rs `292cdffe`) plus nsys profiles on the same
container.

## Decisions

- The already-merged view-materialization fix (`TensorDot::dot_general_read`
  contracting strided reads instead of copying them through
  `to_contiguous_read`) stays as the first step: it removed 5491 cuTENSOR
  permute kernels (346 ms of GPU time, 42% of eager GPU time) from
  `gpu/tensornetwork` eager and 39% of the batched-matmul call time.
- Small-size GPU rows (n=2, 4, 8) are latency/overhead diagnostics, not GPU
  throughput: device kernels are ~0.1 ms of a 0.5-2.6 ms call. They were split
  out of `gpu/linalg_jvp_vjp` into `gpu/linalg_ad_latency` in the benchmark
  repository so the regime is explicit.
- Host-side per-instruction work inside prepared execution is not the remaining
  bottleneck: a per-phase probe measured ~11 us per instruction (10.2 us
  dispatch, ~2 us for staging, validation, and retention combined) while the
  measured call takes ~76 ms.
- Rejected: eliding same-domain `cuStreamWaitEvent` dependencies. All 6982
  sampled node dependencies are same-domain and therefore already ordered by
  stream semantics, but skipping them changed the median by 0.4% (76.15 ->
  75.85 ms), so they are not the gap source.
- Rejected as the cause: tenferro's own CUDA context management. An
  attribution probe over every context and event site counted 27923 of 27924
  `set_current_cuda_context` calls on the fast path and zero `RawContextRestore`
  guard uses in this workload; the per-kernel `cuCtxSetCurrent` traffic observed
  in nsys comes from cubecl-cuda's per-command `unsafe_set_current`
  (`crates/cubecl-cuda/src/compute/server.rs`, with an upstream TODO for the
  missing same-thread check) and costs an estimated 1.5-4.5 ms per call.
- Chosen direction for the dominant gap: event-based cuTENSOR workspace
  retirement, replacing the stream barrier in `Workspace::drop`. Implemented as
  `WorkspaceRetirementQueue` (drain before allocating a workspace, on explicit
  `synchronize()`, and at teardown; capacity 16 with an event-wait then
  stream-barrier fallback that leaks rather than races). See
  `docs/design/gpu-workspace-retirement.md`.

## Verification conclusions and constraints

- `gpu/tensornetwork` trace (post materialization fix): 503 kernels per call,
  35.8 ms kernel busy, 78.5 ms span, 42.7 ms inter-kernel gaps (median gap
  32.5 us). The gaps, not the kernels, are what is left to recover.
- Deferred retirement result: `gpu/tensornetwork` trace 75.7-80.1 ms ->
  68.0-69.5 ms and eager 81.0-83.5 ms -> 72.8-74.5 ms over three repetitions;
  `cudaStreamSynchronize` 2121 -> 27 calls with 162 `cuEventSynchronize` waits
  (0.02 per kernel); median inter-kernel gap 32.5 us -> 28.4 us with unchanged
  kernel busy time. The full `nvidia-gpu` set reported 156 ok / 51 unsupported /
  0 failures and the ignored CUDA library tests 136 passed.
- Remaining 28.4 us median gap is not retirement. Next candidates: cubecl-cuda's
  per-command `cuCtxSetCurrent` and launch latency itself.
- Command-count finding: 4706 of 7215 kernels in the same profile are
  `broadcast_multiply` (~362 per call, ~2.3 us each), while consecutive launch
  submissions are a median 54.1 us apart and the driver launch itself is
  4.1-10.8 us. A prepared-program census settled where they come from: the
  compiled graph holds 549 instructions and all of them are FFI einsum extension
  ops (host=0, other=0), so `segment_exec_program` has no fusion candidates. The
  broadcast multiplies are emitted by tenferro-einsum's own execution
  (`outer_product`), already as one fused command per node through
  `BackendSession::execute_broadcast_multiply`. Prepared-path fusion was designed
  on the opposite premise and is marked superseded in
  `docs/design/gpu-prepared-elementwise-fusion.md`; the open question (can the
  per-node broadcast-multiply be avoided at all) belongs to tenferro-einsum and
  needs its own measurement.
- Barrier attribution: `synchronize_raw_stream` runs 2944 times per run
  (~226 per call); its only non-explicit caller is `Workspace::drop`, reached
  from `CachedCutensorContraction` eviction. Raising the plan cache from 64 to
  512 entries cut the barrier count by 33% (2944 -> 1963) without improving the
  median in a single A/B, so cache sizing alone is not the fix and the A/B needs
  repetition before drawing conclusions.
- cuTENSOR contraction specs are identical between the trace and eager paths
  (74 unique specs, 2340 calls each), so the remaining differences are layout
  materialization, retirement barriers, and instruction count (51 vs 35 kernels
  per call against PyTorch on `grad_sum_qr_jvp` 2x2).
- Not verified: whether the remaining inter-kernel gap can be reduced further,
  and whether the same retirement reasoning should cover the cuFFT work area and
  cuTENSOR permutation plans.
