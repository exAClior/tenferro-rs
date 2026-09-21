# Erase the preset variant enum into an opaque value (#1810)

## Decisions

- `tenferro_tensor_core::DefaultScalars` is now an opaque struct over a private
  inline payload enum (`DefaultScalarsValue`), instead of a public seven-variant
  enum. The preset variants are no longer public API; `DefaultScalars::F64(..)`,
  `use DefaultScalars::F64;`, `use DefaultScalars::*`, and `Tensor::F64(..)` no
  longer compile. Construction and reads go through
  `DefaultScalars::from_vec_col_major` (or `TensorScalar::into_tensor`) and the
  typed accessors `as_slice` / `as_mut_slice` / `into_vec_col_major` / `as_view`.
- `pub use DefaultScalars as Tensor;` stays: `Tensor` is this crate's public name
  for the dynamic host tensor, and the issue's "erase the type" applies to the
  variant list, not to the name. `tenferro_tensor::core::Tensor` is unchanged.
- The payload stays **inline**. The issue's alternative shape - an erased payload
  such as `ErasedHostTensor` - was rejected on measurement: it grows the value
  from 104 B to 216 B, adds one allocation per value (the internal `Arc`), and
  turns the derived deep-copy `Clone` into a shared handle, while
  `docs/design/scalar-composition.md` already recorded the same trade for the
  sibling `Tensor` and chose inline: "the box ... is paid for with a per-tensor
  allocation outside the accounted pool ... **Decision.** The removal takes the
  **inline** shape."
- The private per-variant matches remain (they are what an inline heterogeneous
  payload costs). Deleting them needs the erased-with-allocation payload above,
  so they are kept and the public surface is the thing that shrank.

## Verification conclusions and constraints

- Measured before and after: `size_of::<DefaultScalars>()` is 104 B, construction
  of a value from a host tensor adds no allocation beyond the host tensor's own,
  and `Clone` still deep-copies the buffer. The private payload keeps the
  `Clone` / `Debug` / `PartialEq` derives, so the public wrapper's derived traits
  behave exactly as the enum's did. `the_opaque_value_keeps_the_typed_access_paths`
  pins the surface and the size; `the_default_set_reports_every_member` drives
  every preset dtype through the constructor and the tag.
- `ScalarSet::{TAGS, tag, promote}` and `DType` are unchanged, so
  `crates/tenferro-tensor/src/validate/mod.rs` and the `ext/` sets keep working.
  `ScalarSet` itself is unchanged; downstream sets still declare payload-carrying
  value enums with `define_scalar_set!` (only the default set is opaque).
- `crates/tenferro-tensor-core/tests/scalar_set_import_forms.rs` now documents the
  supported forms for the opaque value: the `Tensor` re-export, the
  `DefaultScalars` name, and the tag.
- `ext/df64-proof/tests/composition.rs`'s cross-set identity demonstration reads
  the default set's payload through `into_vec_col_major::<f64>` and rebuilds the
  host tensor, so the demonstration keeps its meaning without naming a variant.
- Semver: breaking for `tenferro-tensor-core` (preset variants removed). The
  workspace's only users were this crate's tests and doctests plus the
  `ext/df64-proof` test; all were migrated in the same change. `cargo test -j 16
  --workspace` and `cargo clippy --workspace --all-targets -- -D warnings` pass.
- Still open in #1810's text: nothing about the value's *representation*; if the
  scalar-composition program later wants a boxed or tag-only payload, that is a
  new decision on top of this shape.
