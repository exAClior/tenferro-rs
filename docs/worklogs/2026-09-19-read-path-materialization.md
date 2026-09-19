# Read-path materialization

## Decisions

- Audit CPU and CUDA dispatch together. A borrowed view is not evidence that packing is necessary. The existing `Materialization And Copies` rules will name owned/read/output-reuse parity and duplicate packing explicitly rather than adding another audit framework.
- Preserve read descriptors through capable execution boundaries. Owned arguments borrow into the same read path; allocating operations own their final output, and output-reuse operations write to the supplied destination. Never fabricate a new owning tensor over borrowed storage.
- Keep required ownership copies, destructive vendor work buffers, dtype conversion, and layout packing explicit. Packing a noncompact input and then copying it into a destructive work buffer should eventually be one preparation operation, not two independent layers.
- CPU already has direct strided elementwise/read-into and Faer linalg read paths. Common extension dispatch currently bypasses several linalg read hooks; route those operations through the existing hooks rather than inventing another interface. BLAS-only/owned-only operations remain a separately identified boundary until their provider contracts are changed and measured.
- CUDA core read adapters and reduction initialization are the immediate measured target: the existing GPU chain profile shows forty extra materialization kernels per expression. Broader FFT/linalg/indexing changes need their own focused evidence; do not claim those gaps are solved by a core adapter fix.

## Verification conclusions and constraints

- Common linalg extension dispatch now reaches the existing read hooks for SVD/values, QR, rank-revealing QR, eigh/values, LU, full-pivot LU and eig, retaining gauge processing and typed errors. CPU Faer passed 140 library tests and 247 autodiff integration tests, plus all-target clippy with warnings denied. The routing regression fails against the original dispatch and passes after the change; a strided-input numerical test also checks input preservation. No CPU speedup is claimed.
- CPU is not universally affected: ordinary elementwise/read-into already has native strided replay, and FFT has a compact-read override. CPU BLAS linalg view adapters, noneligible CPU read-into fallbacks, and common owned-only indexing/conversion remain explicit audit findings; the present Faer dispatch correction does not eliminate them.
- The same 25 focused extension tests also pass with OpenBLAS and default features disabled; the BLAS fallback behavior remains supported. OpenBLAS was found at `/opt/openblas` and its effective thread count was checked as one.
- CUDA directly launches existing kernels for zero-offset column-major compact reads: add/sub/mul, float/complex divide, neg, and real analytic unary operations including tanh. Same-shape owned add/sub/mul/neg `_into` writes prepared destination storage. Nonempty owned reductions launch the first axis from the input; empty axes still return independent storage.
- Main integration corrected the draft's CubeCL launcher spelling, added the missing unary `_into` destination-residency check, and scoped the reduction contract assertion to nonempty axes. CUDA all-target clippy passes with warnings denied. 159 ignored device unit tests passed, followed by an additional compact/transpose/offset/negative-stride and host-destination regression. 99 non-device unit tests passed. Of 73 integration contract tests, 72 passed initially; the corrected empty-axis assertion passed on rerun. All 28 CUDA linalg integration tests passed. Formatting, crate/linalg boundaries, public error documentation, and storage ownership checks passed. The separate scalar-composition report checker was not run: it requires a generated report and is not evidence for these changes.
- This is a scoped repair, not universal path unification. Noncompact/offset elementwise reads, scalar/broadcast binary inputs, unlisted operations, view destinations and most `_into` variants still use existing fallbacks. CUDA FFT/linalg packing, structural reads, and read-reduction view packing are not removed. Existing raw access preparation is not treated as supporting arbitrary view offsets.

## Paired performance and data-movement evidence

Artifacts are under the separate benchmark worktree's `data/diagnostics/read-path-results/`; the predeclared protocol is `data/diagnostics/read-path-protocol.md`. `candidate.patch` and executable SHA-256 hashes identify the measured implementation. Published reports are unchanged.

- Separate Nsight profiles of ten eager 1m chains (3 warmups + 7 timed runs) retain 80 add, 80 multiply and 80 tanh kernels on both versions. `tensor_permute` decreases from 403 to 3: **40 materialization kernels per expression are removed**, leaving three one-time/setup copies. `verify-copy-trace.py` checks the counts; profiled times are not used below.
- Three fresh-process pairs per backend, 5 warmups/30 samples, explicit 1T, sequential execution and the declared reversed second-round order. Idle GPU utilization was 0%, with no other compute process. All 36 records pass numerical verification. Medians and IQRs are retained in `paired-summary.json` and the raw JSONLs.

| Round | Eager 1m baseline ms | Candidate ms | Candidate/baseline |
| --- | ---: | ---: | ---: |
| 1 | 1.663575 | 0.538593 | 0.324 |
| 2 | 1.049020 | 0.445447 | 0.425 |
| 3 | 1.501090 | 0.586358 | 0.391 |

All three ratios satisfy the predeclared <=0.90 gate: approximately 2.35–3.09x faster for this eager workload. Small eager chain ratios are 0.206–0.323. Trace and BMM ratios satisfy the <=1.10 nonregression diagnostic in every round; BMM ratios span 0.983–1.024. This does not establish a general backend speedup. These measurements isolate the materialization repair and do not include the separately investigated CubeCL notification change.

## Dependency integration

CubeCL's notification fix merged in [PR #19](https://github.com/tensor4all/cubecl/pull/19), commit `a2adda17affd40494393a1f40d90980e1235617c`, with all six final hosted checks passing. The fork CI needed portable runners and direct Miri package selection to avoid a silent all-test skip; final Miri logs show 43 actual passing tests including the five wakeup regressions. The user accepts the earlier small-case GPU latency regressions as documented limitations, without changing the historical failed gate or weakening correctness/safety validation. No application CPU affinity is imposed.

Cubek must share the same CubeCL git revision to preserve runtime type identity. Its existing 0.2 release line was updated in [PR #13](https://github.com/tensor4all/cubek/pull/13), merged as `739dbfb414dffb27d44af3ec369f1e8bead9cccf`; the unrelated 0.3 development line and local dependency overrides are not used. Cubek's five CI checks pass, but its existing Linux policy compiles WGPU tests without executing them; six local standard-library tests did execute successfully.

Final integration with both merged pins passes 160 CUDA device unit tests, 99 non-device unit tests, 73 GPU integration contracts and 29 CUDA linalg integration tests. The repository fast PR gate passes, including its root/tropical/sparse formatting and Clippy profiles plus 25 focused linalg extension tests. Additional all-target GPU Clippy with both CUDA and WebGPU features passes, and Cargo resolves one CubeCL common revision. The standalone CubeCL sample and pin-contract tests use the same revision; the sample compile-check and five CUDA storage identity/binding tests pass. The retained-address and stream-slot invariants were rechecked against the dependency's OOM-reclaim changes. The full eight-suite benchmark refresh will use the merged tenferro commit rather than this worktree.

Complete tenferro workspace/provider/coverage/CI matrices remain hosted-CI obligations; WebGPU runtime execution was not checked locally.
- Preserve existing user modifications in the main checkout; implementation uses a separate worktree based on origin/main `40e24634`.
- Host kache has one blob-index metadata error even after repair. Builds must not rely on that cache; the CUDA container has an independent Cargo setup.
