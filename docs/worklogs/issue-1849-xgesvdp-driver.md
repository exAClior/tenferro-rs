# Explicit polar-based CUDA SVD driver (#1849)

## Decisions

- Add `SvdDriver::Xgesvdp` as an explicit alternative, retaining the #1839
  default and CPU behavior. The CUDA integration shares vector and values-only
  workspace handling; wide matrices use the vendor's native support.
- Choose issue option A: discard `h_err_sigma` and document its meaning on the
  variant. Returning diagnostics would expand the output contract beyond this
  feature. No logging dependency or automatic driver policy is introduced.
- Preserve the existing full-SVD public surface, which has no driver option.
  Test its internal full/economy implementation directly rather than add an
  unrelated options API. See [GPU backend design](../design/gpu-backend-design.md).
- Bindings follow the NVIDIA cuSOLVER API reference, cross-checked against
  exAClior's independently written standalone harness supplied with #1849.
  This implementation is offered under `MIT OR Apache-2.0`; it neither copies
  the harness nor derives from NVIDIA's BSD-3 `CUDALibrarySamples/Xgesvdp` sample.

## Verification conclusions and constraints

- Tests cover explicit selection, the unchanged Auto boundary, operation
  identity and values-only pruning, and ignored CPU driver selection.
- Hardware-gated tests cover rectangular and batched ComplexF64, Float64,
  eager/traced execution, values-only execution, reconstruction, orthogonality,
  full factors, and agreement with Gesvd at the issue's acceptance sizes.
- CUDA execution and public-API timing remain pending controlled GPU validation.
  The standalone measurements in [#1849](https://github.com/tensor4all/tenferro-rs/issues/1849)
  are reference evidence, not measurements of this integration.
