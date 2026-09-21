use super::*;

#[test]
fn inline_metadata_collection_keeps_small_shapes_and_strides_inline() {
    let shape = shape_vec(&[2, 3]);
    let strides = stride_vec(&[1, 2]);

    assert_eq!(shape.as_slice(), &[2, 3]);
    assert_eq!(strides.as_slice(), &[1, 2]);
    assert!(!shape.spilled());
    assert!(!strides.spilled());
}

#[test]
fn erased_tensor_size_stays_within_the_documented_bound() {
    // The inline representation is deliberate: an erased `Tensor` must not
    // allocate for its metadata, so its size is the price paid on every move.
    // #1823 recorded 1464 B at its baseline and sliced the duplicated metadata
    // down to 776 B; keep the bound here so a future field addition has to
    // justify the bytes instead of silently regressing them.
    const ERASED_TENSOR_SIZE_BOUND: usize = 776;
    let size = size_of::<Tensor>();
    assert!(
        size <= ERASED_TENSOR_SIZE_BOUND,
        "erased Tensor grew to {size} B, above the {ERASED_TENSOR_SIZE_BOUND} B bound"
    );
    assert_eq!(align_of::<Tensor>(), 8);
    // The payload tag lives in a niche, so `Option<Tensor>` costs no extra word.
    assert_eq!(size_of::<Option<Tensor>>(), size);
}
