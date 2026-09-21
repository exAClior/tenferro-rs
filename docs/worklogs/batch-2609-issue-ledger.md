# Batch ledger: #1823 A-G, #1704 (branch `perf/batch-2609`)

Base: `origin/main` `9676f9d6`. Batch head at PR time is listed in the PR body.

Classification vocabulary follows
[`ai/contribution-workflows/repository-remediation.md`](../../ai/contribution-workflows/repository-remediation.md):
`Auto Fix`, `Verify First`, `Stale / Out Of Scope`, `Design Gate`.

## In this batch

| Issue | Class | Evidence from the working tree | Verification |
| --- | --- | --- | --- |
| #1823 A-G | Auto Fix | `size_of::<Tensor>()` measured 1464 B at the base (test profile) with the duplicated metadata the issue names: `smallvec` without `union`, `#[repr(C, u8)]` on `TensorPayload`, `OwnedSpanClaim.root`, `RootBoundSpan.root_identity`, the record's duplicated `span`/`layout`/`dtype`/`element_size`/`root`, and an inline `CheckedLayout::Strided`. All seven slices landed: 1464 -> 1384 (A) -> 1376 (E) -> 1360 (D) -> 1312 (C) -> 1192 (G) -> 928 (B) -> 776 (F). Slice H was filed separately as #1824 and fixed by #1843. | `erased_tensor_size_stays_within_the_documented_bound` bounds the size; `descriptor_plan_allocation_count_separates_contiguous_and_strided` counts 0 allocations for a contiguous descriptor and 1 for a strided one; `cargo test -j 16 -p tenferro-tensor` passes; the trybuild UI diagnostics are byte-identical to pristine `origin/main` after path normalization. |
| #1704 | Auto Fix | `new_leaf` entered one backend session per leaf. `materialize_tensor_read_in_domain` takes the plain host branch whenever `backend_family()` is `None`, so the session did no provider work for an owned host tensor. The eager leaf now states that acceptance and copies through `TensorRead::tensor_view` + `TensorView::duplicate`. | `host_leaf_construction_does_not_enter_a_backend_session` (0 session entries, counter shown to observe real entries); `host_leaf_materialization_matches_the_cpu_backend_acceptance` (accept + view/device/external declines); paired 1-worker release/faer experiment: `from_tensor_in_8` 15.45 us -> 3.92 us (3.95x), controls unchanged (`docs/worklogs/issue-1704-eager-leaf-session.md`). |
| #1823 dependent assertion in `ext/df64-proof` | Auto Fix | `report_erased_payload_allocation_cost` pinned `size_of::<Tensor>() == size_of::<TypedTensor<f64>>() + 8`, which assumed `#[repr(C, u8)]`; the tag now lives in a niche (776 == 776). The assertion was relaxed to the property it exists for (the payload adds at most the discriminant, never a pointer plus allocation). | `cargo test -j 16 -p tenferro-df64-proof` passes. |

## Deliberately not implemented, with current evidence

| Issue | Class | Evidence from the working tree | Why it is not in this PR |
| --- | --- | --- | --- |
| #1810 remove the unused `tenferro_tensor_core::Tensor` variant enum | Design Gate | The issue's premise (no consumers) does not hold on this base. `ext/df64-proof/tests/composition.rs:225` constructs `DefaultScalars::F64(HostTensor<..>)` and matches the variant to demonstrate cross-set identity; `crates/tenferro-tensor/src/validate/mod.rs:60` and its tests use `<DefaultScalars as ScalarSet>::promote`; `crates/tenferro-tensor-core/tests/scalar_set_import_forms.rs` matches `Tensor::F64`. The payload-carrying value enum is also the pattern the `ext/` proof crates demonstrate, and `docs/design/scalar-composition.md` records that this type was deliberately scoped out of the erased-`Tensor` work. Removing it changes a published public type in the middle of the active scalar-composition program (#1785-#1793). | Needs a maintainer decision on direction (remove the payload model, or keep it and drop only the `Tensor` alias), plus its own short design and semver note. Tracked by a comment on #1810. |
| #1838 strided `axpby_read_into_accum` destination | Design Gate | The current destination contract requires a compact injective span in shared validation, the CPU path reaches `strided-basic`'s dense `axpby_accum`, and the pinned `strided-rs` revision (`1be41ce4`) has no strided-destination accumulate; `REPOSITORY_RULES.md` requires adding a missing general primitive to `strided-rs` first. | Deferred by the maintainer to a later batch together with #1615; it crosses a repository/dependency boundary (a `strided-rs` change plus a pin bump). |
| #1615 borrowed conjugated dot reduction | Design Gate | The issue's own process gate requires maintainer acceptance before a new backend operation and provider-inject registration. | Deferred by the maintainer to the same later batch as #1838. |

## Open issues not in this batch

| Issue | Class | Note |
| --- | --- | --- |
| #1628 Mac CPU performance gaps | Stale / Out Of Scope | Its concrete items were filed and closed separately (#1662, #1663, #1668, #1669, #1671, #1672, #1786) or remain tracked by #1765 (FFT lanes) and #1803 (eager backward / batched solve). The report itself measures `11a5b5a3`. Closed with pointers after this PR. |
| #935 guarded shape specialization | Design Gate | Feature request for the fusion path; no accepted design. |
| #974 einsum extension lowering | Design Gate | Feature request; related tracking issues (#975, #1060, #1061, #1237) are closed. |
| #989 graph-level RNG key semantics | Verify First | Design placeholder with no implementation or activity since 2026-06-10; no random ops exist. Candidate to close as stale, but that is a maintainer call, not a remediation edit. |
| #1043 no_std + alloc epic | Design Gate | Multi-phase epic. |
| #1271 layered mechanical audit system | Verify First | Partly delivered: the `// INVARIANT:` convention (#1270/#1272) and checker corrections (#1778, #1782). Phases A1-A5 are not verified complete; needs narrowing before more work. |
| #1658 TAPP contraction boundary RFC | Design Gate | Open RFC. |
| #1660 CPU linalg backend strategy | Design Gate | Long-term strategy. |
| #1695 IR-agnostic AD transform framework | Design Gate | Long-term design. |
| #1765 CPU FFT lane parallelization | Verify First | #1786 landed part of the surrounding work; the parallel-lane claim still needs measurement. |
| #1858 concrete session surface, eager FFT in-place errors, session docs | Design Gate | Filed by the maintainer on 2026-09-21 after this ledger's issue snapshot; it extends #1680 Phase 1 to the remaining operation families and to the two eager in-place FFT methods, so it is public-API convention work across `tenferro-runtime`, `tenferro-ad`, and `tenferro-fft`. |
| #1785, #1787, #1788, #1789, #1790, #1793 scalar composition program | Design Gate | Active program; stage 1 and 2.1 landed in #1800. |
| #1803 residual eager backward / batched solve overhead | Verify First | Active measurement work on M5; unrelated to the leaf path this batch changes. |
| #1848 BLAS/LAPACK symbol resolver | Design Gate | Open design questions in the issue. |
| #1849 `SvdDriver::Xgesvdp` | Verify First | PR #1851 is open. |
| #1852 `syevjBatched` guard | Verify First | Needs a CUDA measurement before changing the heuristic. |

## Post-batch status (2026-09-21, after the merge)

- #1628 closed: its concrete rows are tracked by narrower issues (#1662, #1663, #1668, #1669, #1671, #1672, #1786, #1765, #1803).
- #1765 closed as implemented, with a measurement attached to the issue: the lane loop now runs `(0..jobs).into_par_iter()` inside `session.with_linalg_pool` using `context.native_thread_count()`, and 128^3 3-D c2c scales 43.23 ms (1 worker) to 9.79 ms (8 workers, 4.41x) on this host.
- #1271 narrowed to a phase checklist on the issue: Phase 1 partially landed (marker convention plus categorized baselines), Phase 2 not landed, Phase 3 partial (`api_parity.rs`, `storage_public_api.rs`), Phase 4 not landed (`Cargo.toml` has no `[workspace.lints]`).
- #1810 is open pending the maintainer's representation decision; the measured options (104 B enum today vs ~240 B inline erasure vs 32 B boxed erasure vs 24 B tag-only) and their consumers are recorded on the issue, with a recommendation to close it as wontfix unless the scalar-composition program wants an in-core host value type.
