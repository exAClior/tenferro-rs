use cubecl::stream_id::StreamId;
use std::num::NonZeroUsize;
use std::panic;

use crate::cubecl::dispatch::{
    cubecl_shape_and_strides, typed_tensor_array_arg, typed_tensor_binding,
};
use crate::cubecl::{CudaBackend, CudaExtensionCache};
use crate::{
    CubeclBuffer, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement, StorageBuffer,
    TypedTensor,
};
use tenferro_tensor::{
    AllocationDomainId, BackendStorage, CacheStats, Error, ErrorKind, ValidationError,
    ValidationKind,
};

#[test]
fn cubecl_buffers_keep_domain_and_distinguish_allocations() {
    let domain = AllocationDomainId::fresh();
    let first = CubeclBuffer::new(
        cubecl::server::Handle::new(StreamId::current(), 4),
        4,
        0,
        domain,
    );
    let second = CubeclBuffer::new(
        cubecl::server::Handle::new(StreamId::current(), 4),
        4,
        0,
        domain,
    );

    assert_eq!(first.allocation_domain(), domain);
    assert_eq!(second.allocation_domain(), domain);
    assert_ne!(
        <CubeclBuffer as BackendStorage<f32>>::allocation_id(&first),
        <CubeclBuffer as BackendStorage<f32>>::allocation_id(&second)
    );
}

#[test]
fn scalar_reduction_shape_stays_separate_from_cubecl_launch_metadata() {
    assert!(crate::cubecl::reduction_output_shape(&[2, 3], &[0, 1]).is_empty());
    assert_eq!(cubecl_shape_and_strides(&[]).unwrap(), (vec![1], vec![1]));
}

#[test]
fn cubecl_metadata_uses_dense_column_major_strides() {
    assert_eq!(cubecl_shape_and_strides(&[]).unwrap(), (vec![1], vec![1]));
    assert_eq!(
        cubecl_shape_and_strides(&[2, 3, 4]).unwrap(),
        (vec![2, 3, 4], vec![1, 2, 6])
    );
}

#[test]
fn cuda_extension_cache_is_type_indexed_and_lazy() {
    let cache = CudaExtensionCache::new();
    let mut initializers = 0usize;

    {
        let value = cache
            .get_or_try_init::<usize>(|| {
                initializers += 1;
                Ok(17)
            })
            .unwrap();
        assert_eq!(*value, 17);
    }
    {
        let value = cache
            .get_or_try_init::<usize>(|| {
                initializers += 1;
                Ok(23)
            })
            .unwrap();
        assert_eq!(*value, 17);
    }
    {
        let value = cache
            .get_or_try_init::<String>(|| Ok("gpu".to_string()))
            .unwrap();
        assert_eq!(value.as_str(), "gpu");
    }

    assert_eq!(initializers, 1);
}

#[test]
fn cuda_extension_cache_reports_stats_and_clear() {
    let cache = CudaExtensionCache::new();
    assert_eq!(cache.stats().unwrap().entries, 0);
    assert_eq!(cache.stats().unwrap().retained_bytes, 0);

    let _usize = cache.get_or_try_init::<usize>(|| Ok(17)).unwrap();
    drop(_usize);
    let _usize_again = cache.get_or_try_init::<usize>(|| Ok(23)).unwrap();
    assert_eq!(*_usize_again, 17);
    drop(_usize_again);
    let _string = cache
        .get_or_try_init::<String>(|| Ok("gpu".to_string()))
        .unwrap();
    drop(_string);

    let stats = cache.stats().unwrap();
    assert_eq!(stats.entries, 2);
    assert!(stats.retained_bytes >= std::mem::size_of::<usize>());
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 2);

    cache.clear().unwrap();
    assert!(cache.is_empty().unwrap());
    let stats = cache.stats().unwrap();
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.retained_bytes, 0);
    assert_eq!(stats.clears, 1);
}

#[test]
fn cuda_extension_cache_methods_report_poisoned_lock() {
    let cache = CudaExtensionCache::new();
    let poisoned = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let _guard = cache.inner.lock().unwrap();
        panic!("poison cuda extension cache lock");
    }));
    assert!(poisoned.is_err());

    assert!(cache.is_empty().is_err());
    assert!(cache.stats().is_err());
    assert!(cache.clear().is_err());
    assert!(cache.max_entries().is_err());
    assert!(cache.max_retained_bytes().is_err());
    assert!(cache.get_or_try_init::<usize>(|| Ok(17)).is_err());
}

#[test]
fn cuda_extension_cache_has_configurable_entry_bound() {
    let cache = CudaExtensionCache::with_max_entries(NonZeroUsize::new(1).unwrap());
    let mut usize_initializers = 0usize;

    let value = cache
        .get_or_try_init::<usize>(|| {
            usize_initializers += 1;
            Ok(17)
        })
        .unwrap();
    assert_eq!(*value, 17);
    drop(value);

    let value = cache
        .get_or_try_init::<String>(|| Ok("gpu".to_string()))
        .unwrap();
    assert_eq!(value.as_str(), "gpu");
    drop(value);
    assert_eq!(cache.stats().unwrap().entries, 1);

    let value = cache
        .get_or_try_init::<usize>(|| {
            usize_initializers += 1;
            Ok(23)
        })
        .unwrap();
    assert_eq!(*value, 23);
    assert_eq!(usize_initializers, 2);
}

#[test]
fn cuda_extension_cache_has_configurable_retained_byte_bound() {
    let cache = CudaExtensionCache::with_max_entries(NonZeroUsize::new(8).unwrap());
    let byte_limit = std::mem::size_of::<usize>().max(std::mem::size_of::<String>());
    cache
        .set_max_retained_bytes(NonZeroUsize::new(byte_limit).unwrap())
        .unwrap();
    assert_eq!(cache.max_retained_bytes().unwrap().get(), byte_limit,);

    let value = cache.get_or_try_init::<usize>(|| Ok(17)).unwrap();
    assert_eq!(*value, 17);
    drop(value);

    let value = cache
        .get_or_try_init::<String>(|| Ok("gpu".to_string()))
        .unwrap();
    assert_eq!(value.as_str(), "gpu");
    drop(value);

    let stats = cache.stats().unwrap();
    assert_eq!(stats.entries, 1);
    assert!(stats.retained_bytes <= byte_limit);
    assert_eq!(stats.evictions, 1);
}

#[test]
fn cuda_extension_cache_allows_dynamic_retained_byte_updates() {
    let cache = CudaExtensionCache::with_max_entries(NonZeroUsize::new(8).unwrap());
    let value = cache.get_or_try_init::<usize>(|| Ok(17)).unwrap();
    assert_eq!(*value, 17);
    drop(value);

    cache.update_retained_bytes::<usize>(4096).unwrap();
    let stats = cache.stats().unwrap();
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.retained_bytes, 4096);

    cache
        .set_max_retained_bytes(NonZeroUsize::new(1024).unwrap())
        .unwrap();
    assert!(cache.is_empty().unwrap());
    assert_eq!(cache.stats().unwrap().evictions, 1);
}

#[test]
fn cuda_extension_cache_byte_pressure_evicts_largest_entry() {
    let cache = CudaExtensionCache::with_max_entries(NonZeroUsize::new(8).unwrap());
    cache
        .set_max_retained_bytes(NonZeroUsize::new(64).unwrap())
        .unwrap();
    drop(cache.get_or_try_init::<usize>(|| Ok(17)).unwrap());
    drop(
        cache
            .get_or_try_init::<String>(|| Ok("gpu".to_string()))
            .unwrap(),
    );

    cache.update_retained_bytes::<String>(128).unwrap();

    let value = cache.get_or_try_init::<usize>(|| Ok(23)).unwrap();
    assert_eq!(*value, 17);
    drop(value);
    let stats = cache.stats().unwrap();
    assert_eq!(stats.entries, 1);
    assert!(stats.retained_bytes <= 64);
    assert_eq!(stats.evictions, 1);
}

#[test]
fn cuda_extension_cache_resources_survive_plan_pressure() {
    let cache = CudaExtensionCache::with_max_entries(NonZeroUsize::new(2).unwrap());
    cache
        .set_max_retained_bytes(NonZeroUsize::new(64).unwrap())
        .unwrap();
    let mut loads = 0;
    for _ in 0..7 {
        let resource = cache
            .get_or_try_init_resource::<u64>(|| {
                loads += 1;
                Ok(37)
            })
            .unwrap();
        assert_eq!(*resource, 37);
        assert!(cache.inner.try_lock().is_err());
        drop(resource);
        drop(cache.get_or_try_init::<u32>(|| Ok(9)).unwrap());
        cache.update_retained_bytes::<u32>(65).unwrap();
        let stats = cache.stats().unwrap();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.retained_bytes, 8);
    }
    assert_eq!(loads, 1);
    assert_eq!(cache.stats().unwrap().evictions, 7);
}

#[test]
fn cuda_extension_cache_resource_growth_evicts_plans_not_resources() {
    let cache = CudaExtensionCache::new();
    cache
        .set_max_retained_bytes(NonZeroUsize::new(64).unwrap())
        .unwrap();
    drop(cache.get_or_try_init_resource::<u64>(|| Ok(37)).unwrap());
    drop(cache.get_or_try_init::<u32>(|| Ok(9)).unwrap());
    cache.update_retained_bytes::<u32>(48).unwrap();

    cache.update_retained_bytes::<u64>(32).unwrap();

    assert_eq!(
        *cache
            .get_or_try_init_resource::<u64>(|| panic!("resource was evicted"))
            .unwrap(),
        37
    );
    let stats = cache.stats().unwrap();
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.retained_bytes, 32);
    assert_eq!(stats.evictions, 1);
    assert!(cache.get_cloned::<u32>().unwrap().is_none());
}

#[test]
fn cuda_extension_cache_resource_rejections_are_atomic() {
    let cache = CudaExtensionCache::with_max_entries(NonZeroUsize::new(2).unwrap());
    cache
        .set_max_retained_bytes(NonZeroUsize::new(16).unwrap())
        .unwrap();
    drop(cache.get_or_try_init_resource::<u64>(|| Ok(19)).unwrap());
    drop(cache.get_or_try_init_resource::<u32>(|| Ok(23)).unwrap());
    let before = format!("{:?}", cache.stats().unwrap());
    assert!(cache
        .get_or_try_init_resource::<u8>(|| panic!("must reject before init"))
        .is_err());
    assert!(cache
        .set_max_entries(NonZeroUsize::new(1).unwrap())
        .is_err());
    assert!(cache
        .set_max_retained_bytes(NonZeroUsize::new(11).unwrap())
        .is_err());
    assert!(cache.update_retained_bytes::<u32>(9).is_err());
    assert_eq!(format!("{:?}", cache.stats().unwrap()), before);
    assert_eq!(cache.max_entries().unwrap().get(), 2);
    assert_eq!(cache.max_retained_bytes().unwrap().get(), 16);
    assert_eq!(
        *cache.get_or_try_init_resource::<u64>(|| panic!()).unwrap(),
        19
    );
    let bytes = CudaExtensionCache::new();
    bytes
        .set_max_retained_bytes(NonZeroUsize::new(7).unwrap())
        .unwrap();
    assert!(bytes
        .get_or_try_init_resource::<u64>(|| panic!("byte admission"))
        .is_err());
    assert_eq!(bytes.stats().unwrap().misses, 0);
}

#[test]
fn cuda_extension_cache_resource_promotion_preserves_identity_and_bytes() {
    let cache = CudaExtensionCache::new();
    drop(cache.get_or_try_init::<u64>(|| Ok(41)).unwrap());
    cache.update_retained_bytes::<u64>(29).unwrap();
    assert_eq!(
        *cache.get_or_try_init_resource::<u64>(|| panic!()).unwrap(),
        41
    );
    drop(cache.get_or_try_init::<u64>(|| panic!()).unwrap());
    cache
        .set_max_retained_bytes(NonZeroUsize::new(30).unwrap())
        .unwrap();
    assert!(cache.get_or_try_init::<u32>(|| Ok(2)).is_err());
    assert_eq!(cache.stats().unwrap().retained_bytes, 29);
    assert_eq!(
        *cache.get_or_try_init_resource::<u64>(|| panic!()).unwrap(),
        41
    );
}

#[test]
fn cuda_extension_cache_resource_failure_clear_drop_and_owner_isolation() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    struct Resource(Arc<AtomicUsize>);
    impl Drop for Resource {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let drops = Arc::new(AtomicUsize::new(0));
    let cache = CudaExtensionCache::new();
    assert!(cache
        .get_or_try_init_resource::<Resource>(|| Err(crate::Error::runtime_state(
            "test",
            "failed init"
        )))
        .is_err());
    assert!(cache.is_empty().unwrap());
    drop(
        cache
            .get_or_try_init_resource(|| Ok(Resource(drops.clone())))
            .unwrap(),
    );
    cache.clear().unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(
        cache
            .get_or_try_init_resource(|| Ok(Resource(drops.clone())))
            .unwrap(),
    );
    let other = CudaExtensionCache::new();
    drop(
        other
            .get_or_try_init_resource(|| Ok(Resource(drops.clone())))
            .unwrap(),
    );
    drop(cache);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
    assert_eq!(other.stats().unwrap().entries, 1);
    drop(other);
    assert_eq!(drops.load(Ordering::SeqCst), 3);
}

#[test]
fn cuda_backend_exposes_extension_cache_retained_byte_controls() {
    let _getter: fn(&CudaBackend) -> crate::Result<NonZeroUsize> =
        CudaBackend::cuda_extension_cache_max_retained_bytes;
    let _setter: fn(&CudaBackend, NonZeroUsize) -> crate::Result<()> =
        CudaBackend::set_cuda_extension_cache_max_retained_bytes;
    let _cutensor_stats: fn(&CudaBackend) -> crate::Result<CacheStats> =
        CudaBackend::cutensor_plan_cache_stats;
    let _cutensor_getter: fn(&CudaBackend) -> crate::Result<NonZeroUsize> =
        CudaBackend::cutensor_plan_cache_max_entries;
    let _cutensor_setter: fn(&CudaBackend, NonZeroUsize) -> crate::Result<()> =
        CudaBackend::set_cutensor_plan_cache_max_entries;
    let _cutensor_permutation_stats: fn(&CudaBackend) -> crate::Result<CacheStats> =
        CudaBackend::cutensor_permutation_plan_cache_stats;
    let _cutensor_permutation_getter: fn(&CudaBackend) -> crate::Result<NonZeroUsize> =
        CudaBackend::cutensor_permutation_plan_cache_max_entries;
    let _cutensor_permutation_setter: fn(&CudaBackend, NonZeroUsize) -> crate::Result<()> =
        CudaBackend::set_cutensor_permutation_plan_cache_max_entries;
}

#[test]
fn typed_tensor_binding_accepts_valid_backend_metadata() {
    let tensor = cubecl_tensor_with_len(vec![2, 3], 6).unwrap();

    typed_tensor_binding(&tensor, "metadata_test").unwrap();
    typed_tensor_array_arg(&tensor, "metadata_test").unwrap();
}

#[test]
fn from_buffer_col_major_rejects_backend_buffer_len_mismatch() {
    let error = cubecl_tensor_with_len(vec![2, 3], 5).unwrap_err();

    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::ShapeMismatch)
    );
    assert!(matches!(
        error,
        Error::Validation {
            op: "from_buffer_col_major",
            source: ValidationError::ShapeDataLengthMismatch {
                expected: 6,
                actual: 5,
            },
        }
    ));
}

#[test]
fn from_buffer_col_major_rejects_shape_product_overflow() {
    let error = cubecl_tensor_with_len(vec![usize::MAX, 2], 1).unwrap_err();

    assert_eq!(
        error.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );
    assert!(matches!(
        error,
        Error::Validation {
            op: "from_buffer_col_major",
            source: ValidationError::IntegerOverflow,
        }
    ));
}

fn cubecl_tensor_with_len(
    shape: Vec<usize>,
    len: usize,
) -> tenferro_tensor::Result<TypedTensor<f32>> {
    let domain = AllocationDomainId::fresh();
    let handle = cubecl::server::Handle::new(
        StreamId::current(),
        (len * core::mem::size_of::<f32>()) as u64,
    );
    TypedTensor::from_buffer_col_major(
        shape,
        StorageBuffer::Backend(Box::new(CubeclBuffer::new(
            handle,
            len * std::mem::size_of::<f32>(),
            0,
            domain,
        ))),
        Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: 0,
            }),
            cpu_affinity: None,
        },
    )
}
