# CPU eager LAPACK batches

## Decisions

- Start from PR #1820 (`a081f618`), identify repeatable CPU eager/Torch gaps at
  1T before considering 4T, and fix measured local overhead rather than impose
  a Torch-parity target. Preserve the LAPACK algorithms and public contracts.
- Owned batched solve now reuses one LU/pivot scratch and writes into chunks of
  the final RHS copy. Compact borrowed solve copies a slice rather than calling
  checked multidimensional `get` for every element. This removes per-matrix
  tensor construction, allocations and redundant copying, without aliasing
  either input or changing singularity/transpose behavior.
- QR queries workspace once per operation, reuses factor/reflector scratch,
  and constructs only the final Q/R tensors. Real and complex implementations
  share the batch loop. Keep the existing upper-triangle extraction helper;
  introducing a new kernel or cache is unnecessary for these measured gains.
- At 4T, Intel OpenMP workers inherited a single pinned Rayon worker's affinity.
  Actual `/proc` thread masks, not just the reported MKL thread count, exposed
  the problem. Explicit provider placement within the chosen four cores fixes
  it; document this in the CPU execution guide instead of changing executor or
  provider-ownership contracts.

## Verification conclusions and constraints

- Full CPU-ops survey and paired runs cover 194 eager rows (small/large/batched),
  with setup outside timing and bounded output-retaining batches. At 1T,
  2x2/batch1024 solve improved from 2140 to 79 microseconds, QR from 2442 to 237,
  and solve backward from 10133 to 797. With correctly placed 4T workers, the
  corresponding solve/QR medians improved from 2128/2640 to 75/261 microseconds.
  No row in either complete paired run regressed by more than 25%; a complete
  QR-family repeat did not reproduce the initial 21% large-QR slowdown
  (256x256 difference was 2%). These are observations, not acceptance targets.
- Misplaced 4T 1024x1024 GEMM was 116 ms versus 14.6 ms with explicit placement
  in a diagnostic repeat. That is a configuration correction, **not** a claimed
  speedup from the LAPACK code changes. Keep the original 4T measurements as
  placement diagnostics, not four-core throughput evidence.
- Focused tests cover all four LAPACK scalar types, multi-axis batches,
  square/tall/wide QR reconstruction and orthogonality, transpose versus
  adjoint solve, input preservation, empty batches, invalid shapes and late
  singularity. Existing borrowed/strided/output-reuse tests remain applicable.
  The local fast PR gate passed (including clippy, formatting and all 146
  linalg unit tests); 13 batched linalg tests also passed in release mode.
  No AD rules or GPU code changed.
- Remaining small eager/backward gaps are not declared solved: eager VJP still
  analyzes/compiles the source graph and replays a derivative per target, and
  SVD/eigh still use the older batched wrapper helpers. Extending prepared
  multi-output execution across all decomposition/AD families would dominate
  this local fix. No speculative cache or tiny-matrix replacement was added.
- The initial 19-instance einsum survey included per-contraction executor
  re-entry. Correcting the benchmark to use the same untimed shared scope as
  CPU ops reduced the largest 1T gap from 2.46x to 1.44x; at 4T a small chain
  dropped from 495 to 45 microseconds. The largest residual 4T language-model
  path gap is 3.35x, without an established local kernel defect. These are
  benchmark-scope corrections, not additional library speedups. FFT at length
  65536 did not show a 1T eager/Torch regression; at 4T C32 complex FFT is about
  2.6x slower, while RustFFT's single-transform path does not scale like the
  vendor FFT. Short eager FFT cases are explicitly skipped by the benchmark
  timing policy, not reported as fast.

## Experiment identity

Detailed local evidence is retained in the `tenferro-benchmark` worktree
`.worktrees/bench-cpu-eager-1820/data/results/amd-cpu/cpu/eager-1820/`:
`protocol.md`, all CSV/JSONL samples, LAPACK-call attribution, and thread-mask
snapshots. The PR includes the reproduction settings. Benchmark source is
`dd129e7755915fb5439455d3a928beae56c8395c` with two untimed constructor updates
(`Tensor::F64` to `Tensor::from_typed`) for compatibility with #1820.
Candidate CPU code is in `26abf5ec`; rebasing from the original #1820 head onto
its merged main changed only unrelated GPU/CI files.

Measurements use release builds, Linux devcontainer image `05450c4c6ca0`,
AMD EPYC 7713P, CPU1 at 1T and CPUs1–4 at 4T, explicit MKL/OMP/Rayon budgets,
3 warmups and 15 measured CPU-ops batches. tenferro uses installed oneMKL
2026.1; Torch 2.12.0+cpu bundles MKL 2024.2, so the provider family matches but
its version does not. Cross-library ratios include that difference; paired
baseline/candidate comparisons hold the provider fixed. Incidental host noise
is accepted. Local verification uses a sequential MKL preload for its linker
setup; timing binaries link threaded MKL without that preload.
