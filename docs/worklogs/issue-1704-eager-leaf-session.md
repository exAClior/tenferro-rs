# Eager leaf construction without a backend session (#1704)

## Decisions

- `CpuExecSession::to_contiguous_read` already had a session-free path for owned
  host tensors: `materialize_tensor_read_in_domain` takes the plain host branch
  whenever `backend_family()` is `None`, so the session contributed only
  admission, provider exclusion, and session construction. The eager leaf path
  now states that acceptance — owned host payload, host placement, preset scalar
  — and copies through the tensor layer's own view copy
  (`TensorRead::tensor_view` + `TensorView::duplicate`), which is the same
  copying code the CPU backend uses. No new public API was added anywhere: a
  cross-crate `pub` materialization helper in `tenferro-cpu` was written first
  and rejected by the repository rules review as low-level execution reach-through
  for a sibling crate.
- `new_leaf` asks the runtime before entering a session
  (`EagerRuntime::to_contiguous_host_read` →
  `EagerBackend::to_contiguous_host_read`). The CPU and recording backends answer
  from the tensor-layer copy; CUDA and WebGPU answer `None` on purpose, so a host
  tensor handed to a device runtime is still rejected by that backend inside its
  session rather than being silently materialized on the host. Views,
  backend-family buffers, device or managed placements, and caller-owned external
  scalars also keep the session path, where the backend reports its own typed
  error.
- The leaf keeps its two owned allocations (the semantic retained value and the
  `AdValueRecord`). Sharing them would change AD/aliasing isolation that #1776
  records as intentional, so it stays out of this change; the eliminable cost was
  the session, not the duplication.

## Verification conclusions and constraints

- A host eager leaf now makes **zero** backend-session entries, asserted
  deterministically on the recording backend
  (`crates/tenferro-ad/src/eager/tests/leaf_session.rs`) and not from timing. A
  subsequent eager `mul` still enters a session, so the counter is shown to
  observe real entries. `eager_materialization_uses_backend` was updated
  accordingly: after a host leaf the backend materialization count is 0 (it
  previously counted the leaf's own session call), and the strided-view
  materialization count is unchanged.
- `host_leaf_materialization_matches_the_cpu_backend_acceptance` pins the fast
  path against the backend's session entry: an owned host tensor produces the
  same bytes, a transposed view read, a device placement, and a caller-owned
  external scalar are all declined, and the device placement is still refused by
  the CPU backend.
- Frozen one-worker experiment, pinned to one CPU core, release/faer, provider
  default (`cpu-faer`), effective worker count asserted as 1 by the harness:
  `crates/tenferro-ad/benches/eager_leaf_construction.rs`. Baseline is the
  #1823-only batch commit `e85976bf`; candidate is the #1704 commit. Statistic is
  the criterion median, three complete paired runs.

  | case | baseline (us) | candidate (us) | ratio | delta |
  | --- | --- | --- | --- | --- |
  | `from_tensor_in_8` (primary) | 15.454 | 3.917 | 3.95x | -74.7% |
  | `from_tensor_in_256` | 16.032 | 4.194 | 3.82x | -73.8% |
  | `requires_grad_in_8` | 16.056 | 4.167 | 3.85x | -74.0% |
  | `session_open_empty` (control) | 6.518 | 6.661 | 0.98x | +2.2% |
  | `input_tensor_8` (control) | 0.777 | 0.710 | 1.10x | -8.7% |

  Predeclared primary gate: >= 30% improvement on `from_tensor_in_8` → PASS.
  Non-regression gate: controls within 10% → PASS (+2.2%, and the input-tensor
  control improved). Validity: the three baseline medians span 0.9% and the
  candidate medians 1.3%; load average stayed below 11 on 64 cores during the
  runs.
- Context, not part of the gated pair: a single pinned run at the batch base
  `9676f9d6` (`origin/main`) measured `from_tensor_in_8` at 17.188 us and
  `input_tensor_8` at 1.537 us, so the complete #1823 + #1704 batch takes the
  8-element leaf from 17.2 us to 3.9 us on this host. The leaf is now below the
  empty-session floor (6.7 us), which is the expected shape once the session is
  gone.
- Host, provider, and settings: single x86_64 test host (`taskset -c 63`, one
  worker), `cargo bench -j 16`, default `cpu-faer` provider, 200 samples and 5 s
  measurement per case. The numbers in this record are this host's; the issue's
  original measurements were macOS arm64.
- Not eliminated: the remaining ~3.9 us is `RetainedValue` / `TracedTensor` /
  `AdValueRecord` construction plus the host copy of the input buffer. A follow-up
  that shares one allocation between the semantic value and the `AdValueRecord`
  would need an AD/aliasing decision first.
