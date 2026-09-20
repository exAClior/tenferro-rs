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
  `svd`. They were not run in the development environment (no GPU); see the
  PR description for the GPU status.
- The issue's A100 measurements (`gesvdj` 4.2-4.8x slower than `gesvd` on
  400-1024 square `c64` matrices with DMRG-like spectra) motivate the option;
  this change does not alter the default and so does not change those numbers
  for callers that leave `Auto`.
