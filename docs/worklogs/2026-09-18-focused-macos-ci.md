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
