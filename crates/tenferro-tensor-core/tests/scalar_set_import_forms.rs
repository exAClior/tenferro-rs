//! The import and construction forms a downstream crate uses for the default
//! scalar set.
//!
//! The value type is opaque: the preset variants are private, so a downstream
//! crate reaches the payload through the public constructor, the tagged
//! conversion, and the typed accessors. `Tensor` stays the crate's public name
//! for the value type as a re-export of `DefaultScalars`.

mod direct_path {
    use tenferro_tensor_core::{DType, Tensor};

    #[test]
    fn the_reexport_names_the_value_type() {
        let value = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
        assert_eq!(value.dtype(), DType::F64);
        assert_eq!(value.as_slice::<f64>().unwrap(), &[1.0]);
    }
}

mod aliased_path {
    use tenferro_tensor_core::{DefaultScalars, ScalarSet};

    #[test]
    fn both_names_denote_one_type() {
        let via_default = DefaultScalars::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
        let via_tensor: tenferro_tensor_core::Tensor =
            DefaultScalars::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
        assert_eq!(via_default.tag(), via_tensor.tag());
    }
}

mod tagged_path {
    use tenferro_tensor_core::{DType, DefaultScalars, ScalarSet};

    #[test]
    fn the_tag_is_the_dtype() {
        let value = DefaultScalars::from_vec_col_major(vec![1], vec![7_i32]).unwrap();
        assert_eq!(value.tag(), DType::I32);
    }
}
