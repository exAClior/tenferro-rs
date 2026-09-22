# Shared cuTENSOR contraction workspace prototype

## Decisions

- Port the user's frozen rydbergsim-benchmark patch from tenferro v0.5.0
  onto downstream's pinned upstream commit
  [5c67203](https://github.com/tensor4all/tenferro-rs/commit/5c6720328b0d1dc605d6ef6b468ed8c7bb8bc3a3).
  Reuse one workspace per physical stream slot, not per cached plan.
- Preserve #1809's event-based retirement on growth and cache teardown.
  Same-stream ordering and the existing cache mutex protect scratch reuse.
  Keep descriptor/plan ownership and cross-stream input retention unchanged.
- Keep the original prototype's exclusion of scratch from the extension-cache
  byte budget. Counting it there recreates whole-cache eviction. This changes
  the meaning of reported cache bytes and requires maintainer agreement;
  a separate bounded scratch policy/introspection API is outside this port.
- Retain the 64-plan default and existing cache controls. Sharing is automatic
  with the existing `cuda` feature, with no new flag or environment option.
  Zero requests allocate nothing; geometric growth rejects integer overflow.

## Verification conclusions and constraints

- CPU build and tests pass (66 integration tests and one doctest). CUDA-feature
  compilation of all targets and strict clippy pass; CUDA-feature host unit
  tests pass (100 passed, 180 hardware tests ignored). The repository's local
  gate, including workspace/extension clippy, passes in the CPU-only orb.
  The workspace-capacity boundary test also passes in release mode.
  Numerical CUDA regression tests compile but have not been GPU-executed.
- This is an implementation reference, not an independently measured speedup.
  Historical A800 caller counts use code predating #1809. The downstream
  hybrid timing comparison also changes the algorithm and host load, so it
  cannot isolate this patch's effect.
- The shared-workspace design and memory-accounting limitations are recorded
  in [the backend design](../design/gpu-backend-design.md).
