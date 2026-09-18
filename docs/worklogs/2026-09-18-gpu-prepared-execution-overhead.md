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

## Stage 0 status (execution-path parity)

- Numeric parity test: `runtime_prepared_matches_compiled_for_elementwise_chain`
  runs the same chain through `run_compiled` and through
  `prepare_compiled` + `run_prepared` and asserts equal results.
- Plan census and parity matrix: `PreparedCompiledGraph::execution_command_counts`
  reports (prepared commands, segmented commands) as a plan census, and
  `execution_path_command_counts_matrix` asserts the current state: a pure
  elementwise chain is (4, 1) - the divergence Stage 1 removes - while the same
  chain with a reduction is (3, 3), because a run containing a reduction is
  fusion-ineligible in both paths.
- The segmented command count is a plan census, not a runtime submission count:
  it assumes the general elementwise fusion path applies and does not model the
  broadcast-multiply triplet/pair fast paths.
- Real submission counting is deferred to Stage 1 verification. A unit test
  cannot register the CPU engine because tenferro-runtime's dev-dependency on
  tenferro-cpu duplicates the crate (the `EngineRegistration` types differ), and
  `EngineRegistration` does not expose its event domain driver, so a counting
  driver cannot be injected either.
- Correction (caller scan): the segmented executor in `segment.rs` is
  production-dead. Every `ErasedTensorBackendExecutor::execute*` call site is
  inside the `mod tests` module of `runtime/execution.rs`, and the scheduled
  executor is the only production executor for `run_compiled`,
  `run_compiled_values`, `run_prepared`, and scoped/admitted execution. The
  earlier framing ("the unprepared path fuses, the prepared path does not") was
  wrong: both production entry points ran the unfused scheduled executor, and the
  segmented fusion only ever ran in tests. The command census now reports one
  production count, and `execution_path_contract.rs` asserts the segmented entry
  points stay test-only.
- Detection artifacts today: the parity matrix above, plus
  `crates/tenferro-runtime/benches/elementwise_fusion.rs`, which now benchmarks
  `prepared_graph` and `unprepared_graph` in the same group (the earlier
  `segmented_graph` label measured only the prepared path). The `gpu/elementwise`
  benchmark case is still to add.

## Stage 1 status (prepared elementwise regions)

- Regions are planned at prepare time (`runtime::region`) with the shared
  segmentation and eligibility, and the region start node executes the whole
  region as one fused command; covered nodes are skipped and their completions
  resolve to the region token.
- Fallback keeps one region completion: when the backend declines the fusion the
  region dispatches its instructions inside the same enqueue.
- Region inputs that are borrowed views decline the fusion (the fused entry
  point takes owned tensors), so chains over graph inputs keep the
  per-instruction path today; materializing a view for a region is the recorded
  follow-up.
- Evidence: `prepared_elementwise_region_executes_as_one_fused_command` (one
  fused execution above the CPU fused kernel's element floor, results equal to
  the unprepared path) and
  `prepared_elementwise_region_falls_back_when_fusion_is_declined` (one fallback
  below the floor, results equal).
- Stage 2 (single production executor): no production duplicate exists, so the
  remaining work is deleting the test-only segmented executor as separate
  cleanup; the contract test prevents it from returning to production.
- `run_compiled` and `run_prepared` share one prepared program through the
  prepared-entry cache, which the tests isolate by using a second runtime for the
  reference run.

- Stage 1 bench (CPU, `benches/elementwise_fusion.rs`, one idle-machine run, medians):
  prepared vs unprepared `add_mul` 4096 = 85.3 vs 107.1 us, 65536 = 81.0 vs
  107.1 us, 1048576 = 936.6 vs 975.7 us; `broadcast_mul` 256x256 = 958.0 vs
  1080.8 us. The prepared path is now at or ahead of the unprepared path on
  every measured case, where before the region work it was behind on the
  elementwise chains. These are single idle-machine runs, not the three-repetition
  controlled A/B a performance claim needs, and an earlier run under build
  contention was several times slower overall.

- GPU verification with region execution enabled (A100 container, `nvidia-gpu`
  standard suite, run `20260918_region`): 56 ok / 51 unsupported / 0 failures,
  matching the pre-change status set. `gpu/tensornetwork` trace 70.3 ms, eager
  75.1 ms, PyTorch 109.6 ms; dense matmul trace 3.369 ms. The suite is unchanged
  because its prepared programs hold no elementwise graph instructions (the
  tensornetwork program is 549 FFI einsum ops and the linalg AD programs have
  mixed runs), so regions are not planned there. The elementwise-chain benefit
  needs the `gpu/elementwise` case that is still to add.

- Stage 1 correctness coverage added: `prepared_elementwise_region_publishes_multiple_live_outs`
  (one region, two program outputs), `prepared_elementwise_region_stays_separate_from_ffi_op`
  (region covers only the chain next to a matrix multiply), and
  `runtime_compiled_values_matches_prepared_for_elementwise_chain` (value output
  mode keeps its per-instruction path and matches). Note that segments are runs
  of non-host/non-FFI instructions, so a chain followed by a reduction is not
  fusable at all in either path; only an FFI or host boundary ends a region.
- `gpu/elementwise` (new suite, `benchmarks/gpu/elementwise.yaml`) is the GPU
  detection device: chain `tanh(t * a + b)` x 8 at n=1024 and n=1048576. At
  292cdffe the trace rows read 1.047 ms and 1.011 ms against PyTorch's 0.154 ms
  and 0.324 ms; with the region work they read 0.143 ms and 0.190 ms, while the
  eager path (no fusion) stays at 1.586 ms and 1.638 ms.

- `gpu/elementwise` three-repetition A/B (A100, every run collected with
  OMP/MKL/OpenBLAS/Rayon thread settings at 1, recorded in each `run.yaml`):

  | case | tenferro-rs 292cdffe | region work (3 reps) | PyTorch (3 reps) |
  | --- | ---: | ---: | ---: |
  | trace n=1024 | 1.047 ms | 0.151 / 0.161 / 0.165 ms | 0.131 / 0.151 / 0.154 ms |
  | trace n=1048576 | 1.011 ms | 0.174 / 0.189 / 0.204 ms | 0.324 / 0.326 / 0.327 ms |
  | eager n=1024 | 1.572 ms | 1.487 / 1.566 / 1.560 ms | - |
  | eager n=1048576 | 1.246 ms | 1.523 / 1.579 / 1.660 ms | - |

  The prepared path is 5-7x faster than before and now ahead of PyTorch on the
  large chain; the eager path, which has no fusion, is unchanged.
- Repeated-run stability: `prepared_elementwise_region_is_stable_across_repeated_runs`
  runs the chain 16 times and asserts each run fuses exactly once with identical
  results, covering the "no accumulating transient state" part of the acceptance
  criteria at the runtime level.

- Stage 1 coverage completed for the acceptance criteria that can be tested
  without fault injection: `prepared_regions_interleave_with_ffi_work` (a chain,
  an FFI matrix multiply, and a second chain in one program: the large chain
  fuses, the small chain after the FFI work falls back below the element floor,
  both outputs match), and `prepared_elementwise_fallback_is_stable_across_repeated_runs`
  (eight fallback runs, one fallback each, identical results). Fault injection for
  delayed completion, partial submit, and event-record failure is still not
  covered; the region executes inside one enqueue, so the existing
  `SubmissionCleanupGuard` synchronizes the stream on those failures.

- Added `enqueue_failure_after_launch_is_not_retried_and_registers_no_completion`,
  a runtime event-domain fault-injection test. Its driver runs the launch once,
  then returns an injected completion-record error; the test verifies that the
  error propagates, no completion token is registered for dependents, and a
  later enqueue does not retry the failed run. This covers the runtime-side
  partial-submit/event-record-failure ordering; CUDA stream cleanup remains
  owned by the backend `SubmissionCleanupGuard`.
