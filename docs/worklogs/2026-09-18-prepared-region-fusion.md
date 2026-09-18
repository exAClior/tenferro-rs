# Prepared elementwise region fusion

Date: 2026-09-18

This worklog records the prepared/unprepared execution-path parity work. The
CUDA workspace-retirement investigation is recorded separately in PR #1809;
this change deliberately does not duplicate that worklog.

## Decision

Use one production scheduled executor for both compiled and prepared runs, and
plan elementwise regions during preparation. A region is executed from its
start node as one fused backend command; covered nodes are skipped and their
completion slots resolve to the region completion token. If the backend declines
fusion, the same enqueue dispatches the original region instructions in order.

The segmentation and eligibility logic is shared with the existing execution
planner. View inputs are materialized once for regions of at least three
instructions; shorter regions use the instruction path. All live-outs and
last-use inputs are retained until the region completion is recorded. Value
mode remains on the instruction path because it does not have the tensor
residency contract required by the fused entry point.

The old segmented executor is production-dead: all of its callers are in
runtime tests. It remains test-only for now; the source-scan contract test
prevents a production caller from returning unnoticed. Deleting it is separate
cleanup rather than a second production executor design.

## Verification

- `runtime_prepared_matches_compiled_for_elementwise_chain` and the parity
  matrix cover numerical equality and command-plan parity.
- Multiple live-outs, FFI neighbours, Value mode, fallback, view materialization,
  and `Fusion -> FFI -> Fusion` interleaving are covered by runtime tests.
- `prepared_elementwise_fallback_is_stable_across_repeated_runs` runs the
  fallback eight times and checks one fallback per run with identical outputs.
- `enqueue_failure_after_launch_is_not_retried_and_registers_no_completion`
  injects a completion-record failure after the launch closure. It verifies one
  launch, error propagation, no completion token for dependents, and no retry.
  Backend stream cleanup remains owned by CUDA's `SubmissionCleanupGuard`;
  delayed-completion fault injection at the CUDA API boundary is not available
  in the CPU test harness.
- The source-scan execution-path contract test asserts that new production
  execution does not bypass the scheduled entry point and that the segmented
  entry points remain test-only.
- The `gpu/elementwise` A/B was collected for three repetitions with effective
  OMP/MKL/OpenBLAS/Rayon thread settings of one. For trace at n=1,048,576, the
  baseline was 1.011 ms and the region path was 0.174/0.189/0.204 ms; PyTorch
  was 0.324/0.326/0.327 ms. The small trace case was 1.047 ms baseline versus
  0.151/0.161/0.165 ms with regions. These are latency-bound detection cases,
  separate from throughput benchmarks.
- The tenferro-runtime local gate passed: formatting, clippy with `-D warnings`,
  and 412 library plus 160 integration/target tests. The pre-existing
  `eager_backend_capability_boundary` trybuild failure remains outside this
  change and is not part of this crate gate.

## Remaining constraints

The runtime counters provide fused/fallback execution counts; a backend-level
raw submission-count harness is still desirable for a future GPU test. The
CUDA cleanup guard is code-reviewed but lacks a fault-injection test for an
actual CUDA event-record failure. The branch is intentionally limited to
prepared-region execution and its regression coverage; workspace retirement and
segmented-executor deletion remain separate follow-ups.
