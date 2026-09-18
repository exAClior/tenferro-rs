# Elementwise fusion in prepared execution

Status: **superseded by measurement. Do not implement as written.**

A census of the prepared program for `gpu/tensornetwork` trace shows why: the
compiled graph contains 549 instructions and **all of them are FFI einsum
extension ops** (`Extension(EinsumExtensionOp { .. })`, one per tree node;
host=0, other=0). `segment_exec_program` therefore finds zero fusion candidates
(candidates=0), and the ~362 `broadcast_multiply` commands per call are not graph
instructions at all: tenferro-einsum emits them from its own execution
(`crates/tenferro-einsum/src/eager.rs` `outer_product`), where the CUDA backend
already serves them as one fused command through
`BackendSession::execute_broadcast_multiply` (`broadcast_multiply_float_e_f32`).

Prepared-path fusion cannot reduce that command count. The remaining question is
different: whether the per-node broadcast-multiply that tenferro-einsum performs
to align operand label orders can be avoided (for example by contracting in the
natural label order and permuting the result afterwards, or by a single fused
broadcast-multiply-contract kernel), and whether that is cheaper than one fused
command per tree node. That question belongs to tenferro-einsum, not to the
runtime's scheduling, and it needs its own measurement before any design.

## Superseded content (kept for the record)

Original premise: `run_prepared` executes one command per scheduled instruction,
the unprepared segmented executor (`crate::segment::eval_exec_segmented_*`)
groups instructions and fuses elementwise runs, and the prepared path should do
the same at prepare time. The measurements below are still valid; the conclusion
that the prepared path holds the fusion opportunity is not.

## Problem (measurements, still valid)

`run_prepared` executes one command per scheduled instruction. The unprepared
segmented executor (`crate::segment::eval_exec_segmented_*`) instead groups
instructions into segments and executes an elementwise run as one fused command
(`segment.rs` → `build_elementwise_fusion_plan` →
`BackendSession::execute_elementwise_fusion`, with dedicated
`broadcast_multiply` triplet/pair fast paths). The prepared path has no fusion:
`crates/tenferro-runtime/src/runtime/execution.rs` contains no reference to it.

That gap dominates what is left of the CUDA gap after the workspace-retirement
fix. In the `gpu/tensornetwork` trace profile (A100, 7215 kernels, 462.8 ms busy,
65.8-66.6 ms span per call):

- 4706 of 7215 kernels are `broadcast_multiply_float_e_f32` (~362 per call), each
  about 2.3 us of device time.
- Elementwise runs are long: run lengths 1:312, 2:364, 3:442, 4:247, 5:104,
  6:91, 7:39, 8:13; 3354 of 4797 elementwise kernels have an elementwise
  neighbour.
- Consecutive launch submissions are a median 54.1 us apart while the driver
  launch itself is 4.1 us (`cuLaunchKernel`) to 10.8 us (`cuLaunchKernelEx`), so
  each extra command costs tens of microseconds of submission time and leaves
  the device idle: 30.6 ms of the 65.8 ms call is inter-kernel gap (46%).

Fusing the median elementwise run of three would remove roughly two thirds of
those commands (~240 per call), worth an estimated 13 ms per call.

## Proposal

Reuse the existing segmentation and fusion machinery in the prepared path, at
prepare time.

1. Run `segment_exec_program` on the `ExecProgram` during preparation, so the
   grouping decision (host, ffi, broadcast-multiply triplet/pair, fused
   elementwise run, and the `segment_outputs_have_future_uses` guard) is made
   once, outside the timed region.
2. Carry the resulting segments in the prepared program, and build the schedule
   over segments instead of instructions: one scheduled operation node per
   segment, with dependencies being the union of its instructions'
   dependencies and outputs being the segment's outputs. A fused segment then
   becomes exactly one command, one completion event, and one retirement unit.
3. Store the `ElementwiseFusionPlan` in the prepared operation for a fused
   segment. Execution calls `execute_elementwise_fusion` for it and falls back
   to per-instruction dispatch when the backend returns `None` (the trait
   already reports an unsupported fusion that way).

Correctness constraints carried over from `build_elementwise_fusion_plan`:
single dtype, single output per instruction, no `BroadcastInDim` source that is
also consumed directly by an elementwise op, and every input and output slot
resolvable to a plan value. Segmentation additionally requires that the
segment's outputs have no future uses outside the segment
(`segment_outputs_have_future_uses`), which is what makes the fused command's
lifetimes match the per-instruction lifetimes it replaces.

## Open decisions

1. Where grouping happens: build the schedule over segments (deeper change to
   preparation and scheduling, but keeps the executor simple and gives one
   completion event per fused command) versus grouping consecutive scheduled
   operation nodes inside `execute_scheduled_slots` (no schedule change, but the
   event-domain enqueue has to be collapsed per group and the plan has to be
   cached in the prepared state). Recommendation: schedule over segments.
2. Fusion eligibility limits: whether to cap the instructions per fused segment,
   and whether to keep the current broadcast-multiply triplet/pair fast paths as
   separate segments or let the general fusion path absorb them.
3. Interaction with slot workspace and `last_use` reclamation: a fused segment
   has one input/output set, so the reclamation of segment inputs must keep the
   existing per-slot `last_use` semantics.
4. Observability: whether to count fused commands (for example through the
   existing prepared-execution stats) so the effect can be verified without
   profiling.

## Verification plan

- Numerical: the CUDA `nvidia-gpu` suites must keep reporting 0 failures with
  the fused path active; add a GPU test that forces a fusable elementwise chain
  through prepared execution and compares against the CPU reference.
- Fallback: a backend that returns `None` from `execute_elementwise_fusion` must
  still execute the segment instruction by instruction.
- Performance: at least three repetitions of `gpu/tensornetwork` trace and eager,
  reporting the median, the command count (kernel count is a proxy), and the
  nsys kernel-busy/span and gap distribution.
