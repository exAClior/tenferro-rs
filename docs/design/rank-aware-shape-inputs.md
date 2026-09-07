# Rank-aware shape inputs

Approved in the design discussion for [#1777](https://github.com/tensor4all/tenferro-rs/issues/1777).

Owned `TypedTensor<T, R>` constructors use `IntoRankShape<R>`. Dynamic `Tensor`
constructors and eager/traced `reshape` use `IntoShapeVec`.

| Input | `Rank<N>` owned constructor | Dynamic rank |
|---|---|---|
| `[usize; K]`, `&[usize; K]` | Accepted only when K = N, at compile time | Any K |
| `Vec<usize>`, `&[usize]`, `ShapeVec` | Runtime `RankMismatch` if length differs | Accepted |

Explicitly borrowing an array as a slice chooses the runtime contract. The
array's dimensions remain runtime values: matching rank does not guarantee
that the shape product matches the data length or fits in `usize`. Existing
checked element-count, layout, and overflow validation remains in force.

`IntoRankShape` lives beside the sealed `TensorRank` contract in tensor-core
and is re-exported by tenferro-tensor. Exact static arrays convert directly;
runtime containers use the existing rank conversion. `TensorRank::Shape`
itself implements the conversion, allowing generic rank-aware callers to
forward their existing shape representation without new call-site bounds.

A blanket conversion through a dynamic vector was rejected because it would
lose compile-time rejection of wrong-length arrays. Additional parallel
constructors and compatibility wrappers are unnecessary. The existing view
constructors' rank-specific shape/stride contracts are not redesigned here.

Executable contracts live in tensor-core's rank doctests and core tests,
`TypedTensor::from_vec_col_major` doctests, tensor's `api_parity` integration
tests, and eager/traced reshape doctests.
