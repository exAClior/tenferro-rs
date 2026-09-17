# Eager read forwarding

The saved provider-matched multiply profile attributes ~5.77M instructions to
input materialization (~14% of the former eager total, before the strided SIMD
fix). `exec_standard_op_on_tensor_reads_in_session` unconditionally copies views
before calling `_read` methods which already accept them. CPU multiply has direct
view dispatch. Default trait `_read` methods actually reject borrowed views;
they do not promise a copy fallback. Production CPU sessions override the read
hooks, and CUDA adapters own their explicit device-local layout handling. Owned-only operations already use `concrete_tensor_read(s)` in
their individual arms. This is meaningful avoidable work, not a helper-only
microbenchmark hypothesis.

Remove only the blanket eager conversion. Preserve dtype promotion and its
necessary conversion copies, owned-only operation boundaries, placement checks,
extension registration, borrowed lifetimes, and backend/session ownership. Do
not change AD rules, storage ownership, or introduce unsafe code. Test with the
existing recording backend, non-contiguous and empty inputs, mixed dtypes and a
backend whose default read hook is unsupported (propagate that error, do not
hide it with a copy); retain eager/AD regression coverage. The initial test
assumption that the default exp_read materializes was disproved by execution
and corrected after inspecting the trait's read_tensor contract. Callers are
internal eager dispatch over the closed CPU/CUDA backend selection, not a new
public generic-backend compatibility promise.

Predeclared instruction protocol: baseline tenferro 57bee76, fixed strided
17e05ff, benchmark 976c4b9 source in the existing isolated benchmark worktree.
Record candidate commit before execution. Use the same OpenBLAS Docker image,
Rust release+debuginfo build, CPU16, explicit and verified 1T. Complete cases:
bin_elementwise_mul_2048x2048, bin_matmul_1024, and
lm_batch_likelihood_sentence_3_12d. Three independent baseline/candidate N1/N3
pairs, one fixed warmup; use median paired percentage changes. Primary multiply
Ir reduction >=20%; each control must regress <=1%. Retain all samples and
inspect GEMM self-cost invariance. Any incomplete run, thread/affinity mismatch,
or numerical failure invalidates the complete experiment. Host workloads and
CPU16's L3 domain are recorded; contention permits instruction diagnostics only,
never native-time promotion. Quiet-host native validation remains separate.

Functional verification: the new recording test fails on the baseline (two
copies, expected zero) and passes after forwarding. 91 unit tests, 354 functional
integration tests and 174 doctests pass. The UI integration wrapper is not green:
trybuild initially loses command-line dependency patches; temporary manifest
patches allow compilation but two snapshots differ only in diagnostic underline
and remapped source-path formatting. The manifest was restored byte-for-byte;
no unrelated snapshots were blessed. This remains a final-integration blocker,
not a numerical failure or a claimed fully passing repository gate.

Integration/publication and dependency pin changes are explicitly deferred until
the optimization work is collected, per the user's instruction.
