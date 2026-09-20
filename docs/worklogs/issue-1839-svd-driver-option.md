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

- The `select_svd_driver` policy is unit-tested without a GPU: an explicit
  driver wins at every size, and `Auto` reproduces the 1024 threshold on both
  sides. The GPU source-contract test pins that `svd_read`,
  `svd_with_options_read`, and the values-only hooks all reach the same
  cuSOLVER dispatch with the caller's driver.
- The `#[ignore]` CUDA integration tests in `tests/integration/gpu_linalg.rs`
  force `gesvd` at 64x64 / 8x64 and `gesvdj` at 1025x8 / 8x1025, check
  reconstruction and CPU agreement through the owned, borrowed, and
  values-only entry points, and check `Auto` matches `svd`. They passed on an
  NVIDIA A100-SXM4-80GB (driver 580.173.02, CUDA 12.8, cuTENSOR 2.6.0,
  rustc 1.96.0) together with the pre-existing CUDA SVD tests; the borrowed
  values-only assertion was added after that run and is compile-checked only.
  Hosted CI does not run these tests on every PR.
- The issue's acceptance criterion holds through the public API on that host:
  for an 800x800 ComplexF64 matrix whose singular values span ten decades,
  `SvdDriver::Gesvd` takes about 127 ms against about 600 ms for `Auto` and
  `Gesvdj`, with singular-value error near 1e-14 (Jacobi: near 1e-12).
  `Auto` timings coincide with `Gesvdj` up to 1024 and with `Gesvd` at 1200,
  so the default is unchanged. Complete timings and the reproducer are in
  [issue #1839](https://github.com/tensor4all/tenferro-rs/issues/1839); the
  gain is spectrum dependent (about 1.8x on Gaussian inputs), so no default
  change is proposed.
- Constraints: `tenferro-linalg` compiles GPU linear algebra only under the
  `cuda` feature, so the driver has no other device backend to reach;
  `Xgesvdp` was not implemented.
