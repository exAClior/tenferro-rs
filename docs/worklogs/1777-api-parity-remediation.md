# API parity and audit-rule remediation (#1777)

## Basis and scope

The source report is [#1777](https://github.com/tensor4all/tenferro-rs/issues/1777),
originally audited at `e5b8c65ee6b1e2c418b1b23e5b91feb6dc68893c`.
Integration starts from `a3af5b044f9a970c44f813569386126240426df0` on main,
not the audit checkout's unrelated eager-extension implementation branch.
The original checkout's local CPU arbiter change and `HANDOFF-1597.md` remain
outside this remediation worktree.

The batch covers matrix norms, norm axis validation, typed device-access error
sources, shape input contracts, checker accuracy, agreed repository-policy
changes, and associated public documentation. It adds no backend, dependency,
second audit framework, performance project, or symbolic norm implementation.

## Maintainer decisions

The interactive design discussion approved these boundaries before implementation:

- Preserve static array-rank checking: an owned `[usize; N]` is accepted for
  `Rank<N>`, not a different static rank. Vectors and slices are checked at
  runtime. Dynamic rank accepts arrays regardless of inline storage capacity.
  Explicit conversion of an array to a slice selects the runtime contract.
- Remove the mandatory final six-lane plus integration-auditor protocol and
  its routing/contract-test requirements. Preserve ordinary self-review,
  relevant tests, required CI, and optional explicitly requested audits.
- Retain the existing false-positive marker/source-contract-test requirements
  and their applicability conditions. This intentional local rule prevents
  an audit bot from repeatedly rediscovering the same false positive. Fixing
  a checker's false positives is distinct from weakening this rule.
- Retain the default 90% line-coverage policy, explicitly linking its existing
  numerically verified linalg AD exceptions rather than changing thresholds
  or adding tests solely to pad line coverage.

## Implementation and verification record

- Norm uses original signed/complex inputs for singular-value branches and
  shares axis validation before shortcuts. Owned/read/typed/eager/traced
  regressions include both orders, empty/nonempty duplicate axes, strided
  views, axis permutation, and keepdim. Existing Frobenius/AD tests remain.
- Device preparation preserves its original typed source through nested
  wrappers. Host read/write and view paths downcast the original `AccessError`.
- Rank-aware owned constructors and dynamic eager/traced reshape implement the
  approved shape contract. Clippy's nine obsolete array borrows in existing
  callers were removed without changing their values or expected behavior.
- Boundary checks reuse the existing audit Rust lexer and parse manifest
  dependencies (including aliases and target tables). Error-doc scanning uses
  tracked `crates/` and `ext/` Rust files. Facade checks skip nested worktrees
  and historical documents. API inventory now includes all three standalone
  extensions: 17 library crates, versus 14 before. Its seven pre-existing
  lexical convention candidates remain advisory; this inventory is not a
  compiler-resolved public-API certification.
- The mandatory final audit section, routing registration, and ceremony test
  were removed. False-positive marker/test wording remains, with an explicit
  local-override explanation. Existing AD coverage exceptions are linked.
- TensorValue examples demonstrate ownership, layout views, and recovery after
  a failed consuming reshape. Norm docs describe immediate symbolic rejection
  accurately. The facade build command now names `tenferro-tensor`.

Local evidence:

- `cargo test -p tenferro-tensor --test api_parity`: 2 passed.
- `cargo test -p tenferro-tensor --lib`: 275 passed.
- `cargo test -p tenferro-tensor --doc`: 374 passed, including constructor
  compile-failure and TensorValue examples.
- `cargo test -p tenferro-tensor-core`: all three suites passed (core tests,
  typed errors, and 89 doctests).
- `cargo test -p tenferro-ad --doc 'eager_ops::EagerTensor::reshape'`: 1 passed;
  runtime's traced reshape doctest filter: 2 passed.
- `cargo test -p tenferro-linalg --features autodiff --test integration norm`:
  21 passed. Numerical regressions use CPU/faer with an explicit one-thread
  backend and assert `num_threads() == 1`; commands set
  `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.
- Boundary scope regression suite: 7 passed; existing error-doc suite: 11
  passed. Public-error-docs, AD boundaries, crate boundaries, no-facade,
  repository-rule tests, documentation-consistency tests, and CI-helper tests
  passed. API inventory generation succeeded with 17 crates.
- `bash scripts/check-pr-fast.sh --coverage-reviewed` with the two focused
  tensor/norm test commands passed, including repository formatting and
  CI-parity clippy for root and standalone extension workspaces.

CPU checks do not establish GPU/XLA success-path coverage. The required hosted
CI matrix, including the GPU gate and coverage, must pass on the submitted PR
state before merge. Existing shipped skill examples retain valid call forms;
no new mirrored API spelling is necessary.
