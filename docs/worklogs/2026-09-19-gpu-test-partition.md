# Separate GPU relevance from device execution

## Decision

Keep conservative PR relevance but move audited device-independent tests from
paid RunPod execution to the hosted preparation job, using the same CUDA/PJRT
feature-enabled archives. An exact checked-in test inventory makes additions,
renames, removals, and duplicate assignments visible rather than silently
classifying future tests by a name or feature flag. Hosted execution is required
even when an archive is restored; it finishes before paid provisioning.

The CUDA inventory contains 897 host cases and 206 single-device cases. PJRT
contains 51 host cases and three CUDA-plugin execution cases. Mixed modules
require individual classification: `determinant_extremes` includes both CPU and
CUDA execution; `storage_provider_cuda` includes a source-only contract.
Type-signature checks referencing `CudaBackend` do not instantiate a device.

Required CUDA tests now assert device availability and are ignored in ordinary
CPU runs. Production availability probing is unchanged. Required PJRT tests
reject a missing plugin. The two-device registration test is explicitly not run
on the existing one-device pod, rather than counted as successful validation.
The maintainer chose to retain the one-GPU provisioning policy; two-GPU
validation remains separate. Two existing trybuild exclusions, the dedicated
A100 benchmark, and three optional external `run_hlo_module` cases retain
explicit separate dispositions. Numerical test bodies are preserved.

## Relevance and alternatives

The Cargo dependency audit includes shared CPU, tensor, runtime, AD, and
extension crates in the CUDA/PJRT/tutorial closure. A GPU-directory-only policy
would be incorrect. Keeping conservative source classification is simpler than
introducing a dependency-policy engine for the few proof/consumer fixtures
outside that closure. Executable `docs/tutorial-code/` must not inherit the
prose-only exception; changes to the partition policy also require GPU CI.

The historical RunPod run 35407000298 spent a summed 24.265 seconds in the
897 host-classified CUDA tests, out of about 130 seconds of CUDA execution.
That is workload placement evidence, **not a measured end-to-end speedup**.
Archive transfer and runtime restoration remain; this change does not claim
smaller archives, faster tutorial builds, or a cache benefit for new source.
Unproven feature-fingerprint tuning and parallel runtime downloads are deferred.

## Verification and limitations

Local execution of the feature-enabled archive passed all 897 CUDA host cases
and all 51 PJRT host cases. With CUDA devices hidden, all 206 selected CUDA
hardware tests failed rather than silently passing. Without a configured PJRT
plugin, all three selected PJRT execution cases failed. These are intentional
negative checks, not physical-device validation. A separate run on the local
A100 80GB PCIe passed all 206 selected CUDA hardware tests using CUDA 12.6;
this is not evidence for the hosted RunPod workflow or the PJRT plugin lane.
Test scheduling and Rayon/OpenBLAS/OMP were explicitly one-threaded.

Inventory tests cover unknown, removed, renamed, duplicate, and wrongly scoped
test identities, malformed membership counts, and explicit not-run reporting.
Workflow contracts retain authorization, prerequisite, cache-trust, serialization,
and cleanup checks and require hosted execution on cache hits before provisioning.

The old trusted-default-branch PR workflow cannot prove the newly edited workflow
wiring. Its physical GPU result and a later trusted post-merge dispatch must be
reported separately. The legacy fork/manual GPU workflow remains unchanged.
