# Execution path parity and prepared-path elementwise fusion

Status: agreed design for the elementwise region work. Stage 0 is implemented as
tests; the production change (Stage 1) is not started. Reviewed by a multi-model
council; the review changed the shape of the plan (Stage 2 is a non-goal for
now, dependency handling is not a union, multi-output regions stay supported).

## Problem and evidence

The two execution paths diverge in how they submit elementwise work.

- Unprepared (`run_compiled*`) segments the `ExecProgram`
  (`segment_exec_program`) and executes a fusable elementwise run as one command
  (`build_elementwise_fusion_plan` -> `BackendSession::execute_elementwise_fusion`).
- Prepared (`prepare_compiled` + `run_prepared`) walks the schedule and
  dispatches one command per instruction; `runtime/execution.rs` has no fusion.

Measured divergence: a pure elementwise chain `y = tanh(exp((x+x) * (x+x)))`
compiles to four elementwise instructions; the prepared path submits four
commands, while `segment_exec_program` groups them into one fusable segment
(census: `fused_segments=1(4 instr)`, `fusable=1(4 instr)`, `consecutive=1`).

Benchmark census of prepared programs: dense/einsum problems are a single FFI
extension op (nothing to fuse); `gpu/tensornetwork` is 549 FFI einsum extension
ops whose `broadcast_multiply` work happens inside tenferro-einsum
(`outer_product`, already one fused command per node); linalg AD
`grad_sum_qr_jvp` 256 has 25 instructions (1 host / 5 ffi / 19 other) forming 5
segments that fusion eligibility rejects because the runs contain `ReduceSum`
and `Transpose`. So this change does not reduce the current benchmark kernel
counts; it removes a general divergence and the per-command cost of elementwise
chains.

Cost context: in the `gpu/tensornetwork` trace profile consecutive launch
submissions are a median 54.1 us apart while the driver launch call itself is
4.1-10.8 us, so each avoidable command is worth tens of microseconds.

## Stage 0 (implemented)

Two tests, no production change:

1. Numeric parity: the same chain through `run_compiled` and through
   `prepare_compiled` + `run_prepared` must produce identical results.
2. Plan census: the prepared program for the chain holds the elementwise
   instructions and `segment_exec_program` yields exactly one fusable segment
   covering all of them, which is the structural statement of the divergence.

Submission counts are not asserted yet; the harness for counting real
submissions per path is a follow-up (the council asked to keep results, plan
census, and submissions distinct).

## Stage 1 (to implement): prepared executes regions

Grouping is decided at prepare time, after placement and dependency resolution,
and the outer schedule topology and boundary contract do not change at run time.

- Reuse the existing segmentation as *candidate* extraction; `Segment::Fused` is
  a candidate classification, not a proof of fusability. Apply one shared
  eligibility check so both paths reach the same conclusion for the same inputs.
- Eligibility: pure elementwise instructions inside one executor, location and
  event domain; never crossing Host, FFI, Transfer, or Barrier.
- Region I/O: enumerate external inputs and **all live-outs** (future uses and
  graph outputs included). Multiple outputs are allowed where the existing
  builder and backend already support them; the earlier single-exit restriction
  is withdrawn. Split at a boundary that keeps unsupported values live; never
  drop an intermediate silently.
- Dependencies: keep the instruction-id to region-node mapping, remove internal
  dependencies, and reconnect external ones. A union of member dependencies is
  not sufficient; the reduced graph must stay a DAG with the required ordering.
- Completion: one region completion must cover every submission the region
  makes, in the fused and the fallback case. Sharing a domain is not by itself a
  witness.
- Slots and lifetimes: stage external inputs and validate every live-out;
  non-materialized intermediate slots are not validated. Derive retention from
  region-boundary liveness instead of reusing per-instruction `last_use`.
  Fallback intermediates, aliased backing storage, and borrowed inputs stay
  alive until their last GPU use. Logical slots may disappear; physical
  allocations may not be released or reused early, on success, mid-run error,
  event-record failure, or unwind.
- Fallback and dispatch: fall back to per-instruction execution only when no
  submission has happened and the input and slot state can be reused safely; do
  not re-execute after a partial-submit error. Reference
  `PreparedOperationPlan` by original instruction id, never by schedule index.
  Do not store run-specific handles or completion tokens in cached plans.

## Stage 2 (non-goal for now)

Making `run_compiled*` a wrapper over prepare+run and deleting the duplicated
executor is not part of this change. It needs its own evaluation of cold-call
cost, caching, ownership, and fast paths.

## Acceptance criteria

- The known four-instruction chain submits exactly one fused command in both
  paths (not merely "the same count").
- Forced fallback dispatches the original four instructions and matches an
  independent oracle.
- Numeric comparison uses the existing per-dtype contract; command count alone
  never implies a new bitwise requirement.
- Multiple live-outs, duplicate returns, and both Tensor and Value output modes
  keep the outputs they need.
- Fusion -> FFI -> Fusion, mixed locations, and ineligible operations keep
  correct dependencies, dispatch, and splitting.
- Delayed completion, mid-fallback failure, and event-record failure never
  reclaim an allocation early, and repeated runs do not accumulate completed
  transient resources.
- Runtime id/epoch and input-signature mismatches keep their current behavior.
- If shared code changes, CPU and any available non-CUDA backend regressions are
  checked; anything not run is reported as not run.
- Existing GPU tests gain no new failures or unsupported results.
- Performance claims record the effective single-worker/thread configuration and
  rest on at least three repetitions with the spread shown.
- Design contracts land in `docs/design`, decisions and verification in a
  worklog.

## Verification layers

- Structure: region formation, all live-outs, dependency reconnection, DAG
  preservation, boundary splitting.
- Numeric and dispatch: independent oracle, broadcast and view inputs, duplicate
  slots, terminal outputs, multiple outputs, FFI interleaving.
- Async: delayed tokens, partial submits, event-record failure, allocator reuse,
  cleanup on error and unwind.
- Real CUDA: one fused submission for an eligible chain, checked separately from
  the kernel launches it produces.

Benchmark: `gpu/elementwise` (pure elementwise chain, one large bandwidth-bound
size and one small per-command size, participants tenferro-cuda-trace,
tenferro-cuda-eager, pytorch-cuda) plus a permanent command-count test, since the
benchmark alone cannot show which path submitted what.

## Open questions to settle before implementation

1. Does the fusion API's `None` path, together with input collection, guarantee
   a state that can be re-executed?
2. On a partial submit or an event-record failure, who owns the inputs and
   intermediate allocations until completion?
3. Does one region completion cover every submission of the fallback path?
4. Do the existing builder and backend really satisfy the multiple-live-out and
   terminal-Value output contracts?
5. Do the signature and epoch checks cover a plan's specialization conditions?

## Non-goals

Adding `ReduceSum`/`Transpose` fusion, tenferro-einsum's internal outer product,
cubecl per-command overhead, AD optimization, CUDA Graphs, new public API, and a
general IR framework. The broadcast-multiply pair/triplet paths and the
single-session strategy must not be deleted or silently slowed down.
