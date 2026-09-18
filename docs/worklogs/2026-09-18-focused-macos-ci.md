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

### Local paired measurements

Baseline `cf971d0f18d0357133da97dae66802e26bd56e2f`; doctest candidate
`a772a01f`. The scalar candidate additionally contains the provider corrections
committed as `93924bb`; those changes preceded its warmup and timed samples.
This measures CI compiler/test latency, not kernel throughput.

EPYC 7713P, rustc 1.97.1, four-CPU affinity (0–3), Cargo jobs 16, four doctest
workers, explicit `CARGO_INCREMENTAL=0`, `RUSTC_WRAPPER=''`, and
`RAYON_NUM_THREADS=OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1` on both sides.
A compiled probe confirmed `CpuBackend::num_threads() == 1`; examples deliberately
demonstrating other thread counts are unchanged. One warmup precedes three
retained samples. The acceptance threshold was >=20% median improvement, with
max/min <=1.30 and load1 <64; both comparisons meet those limits.

| Operation / variant | Sample 1 | Sample 2 | Sample 3 | Median |
| --- | --- | --- | --- | --- |
| Selected faer doctests, edition 2021 | 189.55 s | 190.83 s | 190.74 s | 190.74 s |
| Selected faer doctests, edition 2024 | 12.55 s | 12.55 s | 12.44 s | 12.55 s |
| Scalar evidence, separate dev build | 126.73 s | 128.81 s | 129.32 s | 128.81 s |
| Scalar evidence, ci reuse | 57.07 s | 56.89 s | 56.38 s | 56.89 s |

All **1125 doctests** pass in every sample, including standalone compile-fail
examples: **93.4%** less time for the selected four crates. The scalar comparisons
both pass the object-level assertions: **55.8%** less time. Scalar measurements
model the actual transition from ordinary ci test builds: each sample first
builds the ci test targets, and the baseline's separate dev output is removed.
This does not claim the same saving when dev artifacts are already available.
Selected BLAS doctests also pass. No whole-workspace or whole-PR speedup is
inferred from these selected measurements.

An earlier experiment accidentally inherited local `build.incremental=true`.
It is invalid for promotion, retained under
`/tmp/tenferro-ci-bench/exploratory-incremental/`, and not used above. The corrected
protocol, scripts, verbose compiler commands, timing/load records, reports and
raw outputs are under `/tmp/tenferro-ci-bench/` (`valid-*` and `scalar-*`).

## Verification and limitations

PR #1816's first hosted revision (`a772a01f`) passed Linux faer/BLAS workspace
checks, clippy, rustfmt, the existing trusted GPU lifecycle and coverage.
Coverage ran **3219 tests**, with **227/227 file thresholds** passing, in **5m57s**;
this one run is not a paired coverage speedup measurement. The Linux faer job
was 10m51s, BLAS 8m03s; the GPU gate arrived about 14 minutes after workflow
creation. The new paid-workflow split/toolkit cache is not active on main yet.

That run exposed an actual Apple build failure: `provider-src` unnecessarily
compiled a second Netlib CBLAS requiring gfortran. Supported providers already
export CBLAS, so the redundant dependency/link marker is removed rather than
installing another compiler. The independent feature selection also exposed
missing linalg cpu-blas/WebGPU forwarding and stale Apple tests using pre-session
APIs. Those are corrected without dropping assertions. The exact selected Apple
targets now pass `cargo check --target aarch64-apple-darwin` on Linux, but this
is not native linking or Metal execution. The first docs gate also detected a
stale generated inventory digest; its case inventory is unchanged and regenerated.

Local checks include 1642 relevant nextest tests, 212 CI helper tests, seven CPU
provider-contract tests, CI-parity clippy, formatting, documentation consistency
and actionlint. One final fast-gate invocation had a shell-quoting error in its
test command after clippy passed; the corrected focused nextest command passed.
Coverage now clears only profraw samples before `--no-clean`, retaining builds
without merging stale measurements; the command sequence passes a 1T smoke test.

Native Apple/Metal execution and new GPU/toolkit wall-clock timings remain
pending. Past nextest PASS lines alone are not proof that optional-device tests
executed. No silent hardware skip or threshold reduction is allowed. Local kache
verification still reports one blob-index drift after repair; Rust checks bypass
that wrapper rather than changing the shared cache service.
