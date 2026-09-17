# CUDA cache resource retention

## Decisions

- Keep raw-session resources in the existing backend-owned, mutex-protected
  type map, but outside plan eviction. Plan workspace growth must not repeatedly
  destroy vendor-library handles. Preserve finite entry and logical-byte bounds
  rather than introducing a second unbounded cache.
- Preserve context authority through cleanup and retire initialized streams
  before destruction. On failed destruction, retaining resources and their
  runtime is safer than reclaiming potentially in-flight state. The durable
  contract is in [GPU backend design](../design/gpu-backend-design.md#runtime-cache-ownership).
- Adapt upstream's growing-entry eviction to apply only to plans. Resource
  growth that fits its reserved share must instead evict plans; regression
  coverage distinguishes those paths.

## Verification conclusions and constraints

- The transferred performance experiment used dependency base
  [`42437dd`](https://github.com/exAClior/tenferro-rs/commit/42437ddc4d57d05578835a48b1471ddcb3e657d1)
  plus clean patch SHA-256
  `f91144f4bcad0762e350127e2fb277001847a8ced00941b1788158d13506cefb`.
  The checksum identifies uncommitted source transferred between machines,
  not a substitute for the promoted commit's Git identity.
- That original overlay passed three actual CUDA lifecycle tests and an
  asymmetric complex density-split reconstruction test on an A100. Its
  matched, restored 100-site two-site density-matrix renormalization group
  sweep experiment passed all predeclared correctness and timing gates.
  Detailed workload settings, results, and provenance belong in the submission
  body; this is not a general tensor-workload speedup claim.
- This promotion targets newer upstream code and includes the growing-entry
  adaptation. The original CUDA measurements do not certify this revised
  source. M4Mini has no CUDA device; hardware validation of the promoted
  revision remains necessary.
- The rebased source passes all 19 metadata tests in both non-release and
  release builds with CUDA enabled. The growing-resource regression fails
  before the adaptation and passes afterward. Repository local checks and
  CUDA-feature lint checks pass; actual CUDA lifecycle tests compile only here.
- Workload MWE (`two_site_dmrg_update_reuses_cuda_linalg_handles`): χ=16,
  d=2, complex128 32×32 SVD, eight updates, then 24 distinct `dot_general`s,
  then eight more SVDs. On current main, CUDA 12.8, RunPod L4:
  warmup 1 miss / 7 hits; after plan fill, 0 SVD misses / 8 hits. Identical
  two-site splits already reuse the handle; this sequence does not recreate
  it. The change is the lifetime contract (resource entries stay out of the
  FIFO), not a measured DMRG speedup.
- No vendor mathematics, numerical precision, solver policy, normal-sweep
  synchronization, profiling instrumentation, or dependency pin is changed.
