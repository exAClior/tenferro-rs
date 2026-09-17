# Canonical uninitialized copy dispatch candidate

The earlier compact-operand borrowing candidate was rejected: its 0.020–0.030%
LM instruction improvement missed the preregistered 1% gate. Its complete patch,
profiles and rationale remain in benchmark commit `9c4a580`; none of that
production change is retained.

This candidate changes one non-conjugating CPU layout-copy call from generic
`map_into` to `strided_kernel::copy_into_uninit`. Conjugating copies, provider
fallback/retry rules, ownership, allocation and error-path reclamation are
unchanged. The shared implementation is strided revision
`78d519013e8b44bd80f77eb01d31039f0be9aae6`, selected with local Cargo overrides;
no published dependency pin is advanced.

The shared API reuses existing permutation machinery only for sequential,
non-contiguous f32/f64 copies. It retains map for contiguous, bounded-parallel
and other element types, including complex and padded types. The exact-type
restriction matters because the permutation engine's 4/8-byte paths interpret
storage as native floats. No initialized Rust references are formed over the
unwritten output; successful copying initializes every logical output element.

Kernel-only diagnostic results and nonzero reference checks are committed in
strided-rs-benchmark-suite `47f7c2e`, under
`result/amd-cpu/uninit-copy-kernels/`. The LM-shaped permutation shows 52.265%
fewer instructions; contiguous and tiny controls change by −0.100% and −0.255%
respectively (small regressions, not improvements). These are not wall-clock
claims. Final focused release tests pass 5/5, also under Memcheck with zero
errors; leak checking was disabled.

## Integration validation

Docker Rust1.98.1, shared OpenBLAS0.3.26, no kache; RAYON/OPENBLAS/OMP threads
explicitly 1, test harness threads 1, Cargo jobs 16. Temporary root-manifest
strided patches were restored byte-for-byte after the run. This passes the
patches to trybuild without committing a dependency integration change.

`trybuild` deliberately replaces RUSTFLAGS, so the BLAS-feature positive fixture
initially failed to link cblas symbols. A test-only linker supplies the existing
provider library without changing snapshots or crate behavior:

```sh
#!/bin/sh
exec cc "$@" -L/opt/openblas/lib -lopenblas
```

With `CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER` set to that wrapper and
`RUSTDOCFLAGS='-C linker=/tmp/cpu-openblas-linker'`:

```sh
cargo test -j 16 -p tenferro-cpu -p tenferro-ad \
  --features tenferro-cpu/cpu-blas,tenferro-ad/cpu-blas --no-fail-fast \
  -- --test-threads=1
```

Captured results include AD library91, integration355 (all five UI fixtures),
doctests174; CPU library564 and all integration binaries passed. The outer
client timed out at420 seconds during CPU doctests; inspection confirmed its
owned Cargo/rustdoc descendants were still running. Docker's final `die` event
records **exitCode=0, execDuration=455s**, proving completion of the actual gate.
The stdout tail after the client timeout was not captured (CPU collected219
doctests). Do not misreport the outer command as a synchronous successful run.
Logs and the final container event are retained with benchmark evidence.

## Remaining acceptance

Whole-eager N1/N3 instruction gate: LM reduction >=5%, multiply/GEMM controls
regression <=1%, three matched pairs, same compiler/provider/threads. This gate
was declared before measurement in strided's worklog. Native quiet-host timing,
4T/batch-BLAS combinations, broader AD, formatting/lint and final artifact review
remain required; this is not a completed optimization goal.
