# `SvdOptions::driver` for cuSOLVER SVD selection (issue #1839)

## Decisions

- `SvdDriver { Auto, Gesvdj, Gesvd }` lives next to `SvdGauge` in
  `tenferro-linalg` and is re-exported from the crate root (not the prelude,
  matching `SvdGauge`). `Auto` keeps the JAX-compatible policy unchanged:
  `gesvdj` when both dimensions are at most 1024, otherwise `gesvd`. An
  explicit driver wins regardless of size. `Xgesvdp` stays a follow-up.
- The driver is a field of the traced `Svd` op and is hashed into the payload,
  so a forced-driver SVD is never deduplicated against the default-policy op.
  `SvdVals` also carries the driver: pruning `U`/`Vt` from an `Svd` keeps the
  caller's kernel choice instead of silently reverting to `Auto`.
- `LinalgBackend` gains defaulted hooks `svd_with_options_read`,
  `svd_values_with_driver`, and `svd_values_with_driver_read`. The defaults
  ignore the driver (CPU providers have one SVD kernel); the CUDA session
  overrides them and forwards the driver to the cuSOLVER dispatch. The two
  extension execution paths and `TensorLinalgExt::svd_with_options_read` route
  through these hooks so eager, traced, owned, and borrowed calls all reach
  the same kernel selection.
  - Rejected: threading the driver through `svd`/`svd_read` signatures. That
    would break every existing backend implementation for a CUDA-only knob.
  - Rejected: a thread-local or environment override. It would not be part of
    op identity and would not compose with traced programs.
- The CUDA-internal enum is renamed `CusolverSvdRoutine` so the public
  `SvdDriver` name is not shadowed inside `gpu/linalg.rs`. `svd_full` keeps
  `Auto` because `LinalgOp::SvdFull` has no options surface.

## Verification conclusions and constraints

- Host-only: `cargo test -p tenferro-linalg` (default `cpu-faer`), doctests,
  `cargo test -p tenferro-linalg --features cuda --lib gpu::linalg::tests`
  (the `select_svd_driver` policy tests compile and run without a GPU),
  `cargo check --workspace --tests --benches --examples`, the GPU source
  contract test, doc snippets, fmt, and clippy.
- The new `#[ignore]` CUDA integration tests in
  `tests/integration/gpu_linalg.rs` force `gesvd` at 64x64 / 8x64 and `gesvdj`
  at 1025x8 / 8x1025, check reconstruction and CPU agreement through the
  owned, borrowed, and values-only entry points, and check `Auto` matches
  `svd`.
- GPU run (2026-09-20, NVIDIA A100-SXM4-80GB, driver 580.173.02, CUDA 12.8
  toolkit, cuTENSOR 2.6.0, rustc 1.96.0, commit `10083553`):
  `cargo test -p tenferro-linalg --features cuda --test integration --
  --ignored --test-threads=1 test_cubecl_svd` → 8 passed, 0 failed (the three
  new tests plus the five existing `gesvd`/SVD CUDA tests);
  `cargo test -p tenferro-linalg --features cuda --lib gpu::linalg::tests` →
  2 passed.
- Same host, a standalone reproducer calling
  `svd_with_options_read(SvdOptions::default().driver(d), session)` on an
  `m x m` ComplexF64 matrix with singular values spanning ten decades
  (1 warmup, 5 timed repetitions, spread < 1 %), medians in ms:

  | m | Auto | Gesvdj | Gesvd | max abs Δs / s₁ (Gesvdj / Gesvd) |
  |---:|---:|---:|---:|---|
  | 400 | 205.1 | 205.3 | 53.5 | 3.1e-13 / 4.6e-15 |
  | 800 | 600.3 | 600.2 | 126.7 | 6.1e-13 / 1.0e-14 |
  | 1024 | 866.4 | 866.5 | 203.4 | 8.4e-13 / 9.8e-15 |
  | 1200 | 269.4 | 1260.4 | 269.5 | 9.4e-13 / 1.3e-14 |

  `Auto` matches `Gesvdj` at m ≤ 1024 and `Gesvd` at 1200, so the default is
  unchanged; both overrides take effect on either side of the threshold. On
  Gaussian inputs `Gesvd` was 1.8x faster at every size. These reproduce the
  issue's patched-constant numbers through the public API.
