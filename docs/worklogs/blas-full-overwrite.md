# BLAS full-overwrite candidate

The OpenBLAS-matched diagnostics identified redundant allocated-output zeroing,
including ~4.2M instructions per b32 x 128 batched matrix product. This candidate
reuses the existing uninitialized GEMM witness and faer validation/dispatch
structure for BLAS beta=0. It does not relax destination ownership or form an
initialized Rust slice over unwritten storage. Empty contractions still write
zeros, and unsupported layouts retain the existing initialized fallback.
Canonical packing and its pool-borrow limitations are not changed here.

PyTorch reference: pytorch/pytorch revision
0d62256a2b23365f8e1604297eb23a6545102aa8,
aten/src/ATen/native/{LinearAlgebra,CPUBlas}.cpp. Inspection, not a source port:
CPU bmm uses its native small kernel when M*N*K<400. Other eligible float64
batches use MKL batch GEMM only in MKL builds; the OpenBLAS build loops over
ordinary GEMM. The installed OpenBLAS 0.3.26 has no exported batched GEMM entry.
Tenferro already has feature-gated grouped BLAS batching; merely naming a
system-OpenBLAS loop batched will not provide a vendor batch implementation.
The LM trace includes batches of 1100 independent 12x12x12 products and small
1x11x11 / 1x1x11 products. They cannot simply be merged into one ordinary GEMM.

Validation protocol, before implementation: use the current migrated source
as baseline, holding strided revision 17e05ff and provider OpenBLAS 0.3.26 fixed.
Required correctness: nonzero real/complex products, initialized nonzero-beta
accumulation unchanged, empty contraction, transpose/layout fallback, pooled
reuse, and uninitialized-output memory checking. The instruction experiment
uses sequential 1T execution and complete paired mm1024 / b32-m128-n128-k128
cases: seek >=5% fewer Ir for the batched case, <=1% regression for the control.
Instruction acceptance is not native performance acceptance. Native validation
requires a quiet host; until then the candidate is not promoted as a measured
wall-clock optimization. No new threshold or custom small-GEMM kernel is added
without evidence beyond call counts.
