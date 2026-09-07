use std::error::Error as _;
use tenferro_tensor::{AccessError, Error, ErrorKind, Rank, TypedTensor, ValidationError};

fn assert_original_device_cause(error: Error) {
    assert_eq!(error.kind(), ErrorKind::RuntimeState);
    assert!(error.to_string().starts_with("audit:"));
    let mut source = error.source();
    while let Some(cause) = source {
        if let Some(access) = cause.downcast_ref::<AccessError>() {
            assert!(matches!(access, AccessError::Unsupported { .. }));
            return;
        }
        source = cause.source();
    }
    panic!("original typed AccessError missing from {error:?}");
}

#[test]
fn device_access_wrappers_preserve_original_typed_source() {
    let mut tensor = TypedTensor::<f64>::from_vec_col_major([2], vec![1., 2.]).unwrap();
    assert_original_device_cause(tensor.prepare_device_read("audit").unwrap_err());
    assert_original_device_cause(tensor.prepare_device_write("audit").unwrap_err());
    assert_original_device_cause(tensor.as_view().prepare_device_read("audit").unwrap_err());
    assert_original_device_cause(
        tensor
            .as_view_mut()
            .prepare_device_write("audit")
            .unwrap_err(),
    );
}

#[test]
fn typed_constructor_shape_inputs_keep_static_rank_contract() {
    let array = TypedTensor::<f64, Rank<2>>::from_vec_col_major([1, 2], vec![3., 4.]).unwrap();
    let vector = TypedTensor::<f64, Rank<2>>::from_vec_col_major(vec![1, 2], vec![3., 4.]).unwrap();
    let slice = TypedTensor::<f64, Rank<2>>::from_vec_col_major(&[1, 2][..], vec![3., 4.]).unwrap();
    for tensor in [array, vector, slice] {
        assert_eq!(tensor.shape(), &[1, 2]);
        assert_eq!(tensor.as_slice().unwrap(), &[3., 4.]);
    }
    let error = TypedTensor::<f64, Rank<2>>::from_vec_col_major(vec![2], vec![3., 4.]).unwrap_err();
    assert!(matches!(
        error,
        Error::Validation {
            source: ValidationError::RankMismatch {
                expected: 2,
                actual: 1
            },
            ..
        }
    ));
    let dynamic = TypedTensor::<f64>::from_vec_col_major([1; 9], vec![7.]).unwrap();
    assert_eq!(dynamic.shape(), &[1; 9]);
    assert_eq!(dynamic.as_slice().unwrap(), &[7.]);
}
