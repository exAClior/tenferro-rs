# Focused macOS CI and coverage scheduling

## Decision

Use an Accelerate-only CPU configuration and explicit Apple test targets on
macOS, not the Linux faer workspace profile. Reuse the Apple context, FFT and
shared Cholesky tests through focused Cargo targets. Add a numerical Accelerate
GEMM/Cholesky check that does not require Metal. Do not include the tutorial
package, whose unconditional faer dependency would defeat this selection.

The dedicated lane requires Metal initialization: the old tests could return
success without executing their numerical assertions when Metal was unavailable.
Local/general runs retain their existing optional-device behavior. The required
check name and change-classification/no-op behavior are unchanged.

Coverage retains LLVM instrumentation and per-file thresholds, but uses nextest
for cross-binary test scheduling. Hosted shared profiles print and set CPU-count
build/test parallelism (build cap 16), respecting explicit overrides. No change
to local parallelism, incremental profiles, GPU execution concurrency, GPU trust
or Linux doctest obligations is included.

## Timing evidence and estimate

Completed runs after the cache-path fix:

| Run | macOS job | Coverage job | Linux workspace gate from workflow creation | GPU gate from PR workflow creation |
| --- | --- | --- | --- | --- |
| PR #1811, workspace 35349815812 / CI 35349815872 | 20m55s | 4m42s | 15m15s | 13m48s |
| PR #1809, workspace 35353291446 / CI 35353291402 | 30m06s | 6m38s | 13m17s | 29m06s |
| main, workspace 35358091041 / CI 35358091534 | 17m35s | 6m46s | 15m55s | not applicable |

macOS doctests alone took 11m33s–21m30s in these samples. Linux doctests and
the existing GPU pipeline remain the likely critical paths. Conditional on warm
compatible caches and usable Metal hardware, a provisional macOS target is
3–6 minutes; this is not a measurement of the new configuration. Coverage has
only about 80–120 seconds of test execution available to accelerate, with the
fixture/build/report costs unchanged; budget roughly 3.5–6.5 minutes rather
than expecting an order-of-magnitude gain.

For #1811's timeline, a macOS lane below the other gates would move completion
from 21m14s to about 15m15s. For #1809 it would move 30m27s only to 29m06s,
because the delayed GPU gate dominates. A normally overlapping warm run should
therefore be budgeted around 13–16 minutes, not the sum of all lane durations;
the existing delayed/recovery GPU path can still take around 29 minutes.

## Follow-up: remove repeated compilation and preparation serialization

The approved follow-up combines four changes in this branch: edition 2024 for
`tenferro-ad`, `tenferro-linalg`, `tenferro-runtime` and `tenferro-tensor`; `ci`
profile assembly inspection; independent hosted GPU preparation plus one queued
paid lifecycle; and a trusted-writer, restore-only hosted CUDA toolkit cache.
The edition migration keeps Rust 2021 formatting to avoid unrelated churn.
Compatibility changes make an existing temporary borrow outlive its block and
mark existing exported symbols/LAPACK declarations explicitly unsafe; no numerical
algorithm is changed.

The paid concurrency group retains the old main workflow's identity, so deployment
does not overlap an old running pod with a new lifecycle. Pending requests use
GitHub's `queue: max`, with PR state/head/base revalidation after waiting. Latest
actionlint 1.7.12 does not recognize that supported GitHub key; a one-file,
one-diagnostic exception is paired with explicit queue/lifecycle contract tests.
Cache restores compile PTX with the restored nvcc and headers before use.

### Local paired measurements (initial experiment invalidated)

Baseline `cf971d0f18d0357133da97dae66802e26bd56e2f`; measured candidate
`4e8e705ff72b991f4ff9a597aa99331bb5104c26`. The later BLAS-only extern declaration
fix is not compiled by the measured faer configuration. This is CI compiler/test
latency, not a kernel throughput benchmark. The initial experiment selected
`ci`, but a local `build.incremental=true` override meant it was not the hosted
profile's effective configuration.
EPYC 7713P, rustc 1.97.1, four-CPU affinity (0–3), Cargo jobs 16, four doctest
workers, default CPU backend and OpenBLAS/OpenMP explicitly 1T. A compiled probe
confirmed `CpuBackend::num_threads() == 1`; examples deliberately demonstrating
other thread counts are unchanged. Compiler wrapping is disabled for both sides.
Dependencies are warm; one warmup precedes three retained samples per variant.

| Selected faer doctests | Sample 1 | Sample 2 | Sample 3 | Median |
| --- | --- | --- | --- | --- |
| Edition 2021 | 190.56 s | 189.90 s | 191.79 s | 190.56 s |
| Edition 2024 | 12.54 s | 12.59 s | 12.30 s | 12.54 s |

All **1125 doctests** passed in every sample, including standalone compile-fail
examples. These preliminary times are **INCONCLUSIVE for promotion** because of
the incremental configuration mismatch. The scalar experiment was also stopped;
all preliminary evidence remains in `/tmp/tenferro-ci-bench/exploratory-incremental/`.
The entire paired experiment is repeated with explicit `CARGO_INCREMENTAL=0`
and verbose compiler-command verification. The original acceptance threshold
(>=20% median improvement), three samples, max/min <=1.30 and load1 <64 noise
limits remain unchanged. The same selected BLAS doctests pass. Full protocol,
run scripts and raw results are retained under `/tmp/tenferro-ci-bench/`.
No whole-workspace or whole-PR speedup is inferred from selected doctests.

## Verification and limitations

The selected Apple feature graph contains accelerate-src and no faer. All 206
CI Python tests, Rust formatting, documentation consistency and actionlint pass.
A standalone LLVM/nextest
smoke test verifies the `ci` profile and JSON report CLI combination, not this
repository's full coverage equivalence.

Native Apple execution and new wall-clock timings are not measured on this
Linux host. Cross-target checking stopped in cblas-src because the available C
compiler cannot compile for macOS. Metal availability on the hosted runner must
be verified: past nextest PASS lines alone are not proof that the optional-device
tests executed. No runner replacement, silent hardware skip, threshold reduction,
or additional paid GPU run is part of this change. Local kache verification
still reports one blob-index drift after repair; the Rust checks bypassed the
wrapper rather than modifying the shared cache service.
