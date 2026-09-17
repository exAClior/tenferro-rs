//! Raw-session contract tests (issue #1597).
//!
//! These tests require CUDA hardware and are therefore ignored by default,
//! matching the regular CUDA test convention in this crate.

use crate::cubecl::CudaRuntime;
use crate::cuda::{gpu_available, CudaBackend, GpuExtensionCapability};

use super::*;

fn first_cuda_backend() -> Option<CudaBackend> {
    let devices = crate::cuda::cuda_devices().ok()?;
    let device = devices.first()?;
    CudaBackend::new(device.id()).ok()
}

#[test]
fn raw_session_exposes_stream_and_runtime_identity() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        let identity = session.runtime_identity();
        session
            .with_raw("test.raw_identity", |raw| {
                assert_eq!(raw.runtime_identity(), identity);
                let _stream = raw.stream();
                Ok(())
            })
            .unwrap();
    });
    // Context/device should be best-effort restored after the session; a
    // subsequent backend operation that needs the primary context must still
    // work when restoration succeeds.
    assert!(session_after_raw_still_runs(&mut backend));
}

fn session_after_raw_still_runs(backend: &mut CudaBackend) -> bool {
    backend.runtime().synchronize().is_ok()
}

#[test]
fn raw_session_allocates_output_and_bytes() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        session
            .with_raw("test.raw_alloc", |raw| {
                let _output = raw.alloc_output::<f32>(&[4])?;
                let bytes = raw.alloc_bytes(1024, "test.raw_alloc")?;
                assert!(!bytes.is_empty());
                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn raw_session_reports_capabilities() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        assert!(session.supports(GpuExtensionCapability::CubeClKernel));
        assert!(session.supports(GpuExtensionCapability::NativeModule));
        assert!(session.supports(GpuExtensionCapability::RuntimeCompilation));
        assert!(session.supports(GpuExtensionCapability::RawStream));
        assert!(session.supports(GpuExtensionCapability::SameDeviceAsyncCopy));
        // Peer copy is directional/hardware-dependent at the provider level.
        assert!(!session.supports(GpuExtensionCapability::PeerCopy));
    });
}

#[test]
fn raw_session_restores_context_on_error_path_observably() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        let result: tenferro_tensor::Result<()> = session.with_raw("test.raw_error", |_raw| {
            Err(tenferro_tensor::Error::runtime_state(
                "test.raw_error",
                "intentional failure",
            ))
        });
        assert!(result.is_err());
    });
    // The primary context must still be restorable afterwards.
    assert!(session_after_raw_still_runs(&mut backend));
}

#[test]
fn raw_tensor_ref_carries_validated_span() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    let host = tensor_f32(vec![8], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let gpu = upload(&backend, &host);
    let Tensor::F32(gpu_typed) = &gpu else {
        unreachable!("f32 tensor")
    };
    with_cuda_exec(&mut backend, |session| {
        session
            .with_raw("test.raw_tensor", |raw| {
                let reference = raw.tensor(gpu_typed)?;
                assert_eq!(reference.byte_len(), 8 * std::mem::size_of::<f32>());
                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn raw_retain_tensor_pins_the_allocation_across_a_drop() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    // Clone the runtime handle so the inner scope can upload without
    // re-borrowing `backend` while the mutable session borrow is live.
    let rt = backend.runtime().clone();
    with_cuda_exec(&mut backend, |session| {
        session
            .with_raw("test.raw_retain_tensor", |raw| {
                // Build the tensor inside a scope so it is dropped while the
                // retained guard below stays live.
                let (saved_ptr, retained) = {
                    let host = tensor_f32(vec![4], vec![1.0f32, 2.0, 3.0, 4.0]);
                    let gpu = upload_tensor(&rt, &host).unwrap();
                    let Tensor::F32(gpu_typed) = &gpu else {
                        unreachable!("f32 tensor")
                    };
                    let reference = raw.tensor(gpu_typed)?;
                    // SAFETY: `reference` is a validated span for `gpu_typed`;
                    // copying the pointer value is read-only and the retained
                    // guard keeps the allocation alive past the drop below.
                    let saved_ptr = unsafe { reference.raw_ptr() };
                    let retained = raw.retain_tensor(gpu_typed, "test.raw_retain_tensor")?;
                    (saved_ptr, retained)
                    // `gpu` (and its owning handle refcount) is dropped here.
                };
                let mut retained_ptr = std::ptr::null_mut();
                retained.with_ptr(|ptr| retained_ptr = ptr);
                assert_eq!(
                    saved_ptr, retained_ptr,
                    "retention must keep the allocation alive across the drop"
                );
                assert!(!retained.is_empty());
                // Prove liveness, not just pointer identity: copy the retained
                // allocation into a freshly allocated probe and read it back
                // after the owning tensor is gone. If the guard had let the
                // allocation be reclaimed, the readback would not reproduce
                // the original payload.
                let probe = raw.alloc_output::<f32>(&[4])?;
                let probe_ref = raw.tensor(&probe)?;
                // SAFETY: `dst` is the freshly allocated, uniquely owned probe
                // span; `src` is the retained allocation still pinned by the
                // guard. The copy is stream-ordered, and `synchronize` below
                // completes it before the host readback.
                unsafe {
                    raw.copy_bytes(
                        probe_ref.raw_ptr(),
                        retained_ptr,
                        4 * std::mem::size_of::<f32>(),
                        "test.raw_retain_tensor",
                    )?;
                }
                raw.synchronize()?;
                let back = download_tensor(&rt, &Tensor::F32(probe)).unwrap();
                assert_eq!(
                    back.as_slice::<f32>().unwrap(),
                    &[1.0f32, 2.0, 3.0, 4.0],
                    "retained allocation must still hold the original payload"
                );
                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn raw_retain_tensor_rejects_tensor_from_another_runtime() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    // Upload on a separate runtime instance: its allocation domain differs
    // from the session runtime's, so retention must be rejected.
    let foreign_rt = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).unwrap();
    let host = tensor_f32(vec![4], vec![1.0f32, 2.0, 3.0, 4.0]);
    let gpu = upload_tensor(&foreign_rt, &host).unwrap();
    let Tensor::F32(gpu_typed) = &gpu else {
        unreachable!("f32 tensor")
    };
    with_cuda_exec(&mut backend, |session| {
        session
            .with_raw("test.raw_retain_foreign", |raw| {
                let err = raw
                    .retain_tensor(gpu_typed, "test.raw_retain_foreign")
                    .unwrap_err();
                assert!(
                    matches!(err, crate::Error::RuntimeState { .. }),
                    "foreign-runtime tensor must be rejected: {err}"
                );
                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn raw_resource_guard_is_runtime_scoped_and_type_keyed() {
    if !gpu_available() {
        return;
    }
    let mut backend = first_cuda_backend().expect("CUDA backend should initialize");
    with_cuda_exec(&mut backend, |session| {
        session
            .with_raw("test.raw_resource", |raw| {
                let guard = raw.resource(|| Ok(String::from("cached-value")))?;
                assert_eq!(&**guard, "cached-value");
                Ok(())
            })
            .unwrap();
    });
}

// Real CUDA events and a vendor handle, not a mock destructor. Pointer addresses
// are owned by the cache; access is serialized by its resource guard.
struct RetirementProbe {
    handle: usize,
    context: usize,
    events: std::sync::Mutex<Vec<usize>>,
    result: std::sync::Arc<std::sync::Mutex<Vec<(bool, bool, bool)>>>,
}

impl Drop for RetirementProbe {
    fn drop(&mut self) {
        use cudarc::{cublas::result as blas, driver::result as driver};
        let context_ok =
            driver::ctx::get_current().unwrap().map(|p| p as usize) == Some(self.context);
        let events = self.events.get_mut().unwrap();
        // Check BEFORE vendor destruction: cublasDestroy may itself block,
        // which must not hide missing explicit multi-stream retirement.
        let retired = events
            .iter()
            .all(|&event| unsafe { driver::event::query(event as _).is_ok() });
        let destroyed = unsafe { blas::destroy_handle(self.handle as _).is_ok() };
        for event in events.drain(..) {
            unsafe { driver::event::destroy(event as _).unwrap() };
        }
        self.result
            .lock()
            .unwrap()
            .push((context_ok, retired, destroyed));
    }
}

unsafe extern "C" fn gated_stream_callback(data: *mut std::ffi::c_void) {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    // CUDA owns this transferred Arc until the callback returns. No CUDA API
    // is called from a host callback. Bound the wait even if the test panics.
    let release = unsafe { Arc::from_raw(data as *const AtomicBool) };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !release.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

fn check_resource_retirement(explicit_clear: bool, poison: bool) {
    use cudarc::driver::{result as driver, sys};
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    };
    assert!(
        gpu_available(),
        "this ignored test requires actual CUDA hardware"
    );
    let backend = first_cuda_backend().unwrap();
    assert!(backend.runtime().stream_slot_count() >= 2);
    // Keep a runtime clone until the assertion so RuntimeState::Drop cannot
    // accidentally repair missing cache retirement or context restoration.
    let runtime = backend.runtime().clone();
    let result = Arc::new(Mutex::new(Vec::new()));
    let release = Arc::new(AtomicBool::new(false));
    for slot in 0..2 {
        let mut worker = backend.clone();
        let result = result.clone();
        let release = release.clone();
        std::thread::spawn(move || {
            cubecl::stream_id::StreamId { value: slot }.executes(|| {
                with_cuda_exec(&mut worker, |session| {
                    session
                        .with_raw("test.retained_lifecycle", |raw| {
                            let resource = raw.resource(|| {
                                Ok(RetirementProbe {
                                    handle: cudarc::cublas::result::create_handle().unwrap()
                                        as usize,
                                    context: driver::ctx::get_current().unwrap().unwrap() as usize,
                                    events: Mutex::new(Vec::new()),
                                    result,
                                })
                            })?;
                            let event =
                                driver::event::create(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)
                                    .unwrap();
                            let data = Arc::into_raw(release) as *mut std::ffi::c_void;
                            // SAFETY: raw session owns a live current-context stream;
                            // callback owns its Arc and event lives in the resource.
                            unsafe {
                                let stream = raw.stream().raw_handle() as sys::CUstream;
                                sys::cuLaunchHostFunc(stream, Some(gated_stream_callback), data)
                                    .result()
                                    .unwrap();
                                driver::event::record(event, stream).unwrap();
                                assert!(
                                    driver::event::query(event).is_err(),
                                    "work must still be pending"
                                );
                            }
                            resource.events.lock().unwrap().push(event as usize);
                            Ok(())
                        })
                        .unwrap();
                });
            });
        })
        .join()
        .unwrap();
    }
    if poison {
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = backend.cuda_extension_cache().inner.lock().unwrap();
            panic!("intentional poison before resource retirement");
        }));
        assert!(poisoned.is_err());
        assert!(backend.cuda_extension_cache().clear().is_err());
        assert!(result.lock().unwrap().is_empty());
    }
    let started = Arc::new(AtomicBool::new(false));
    let start_signal = started.clone();
    let retire = std::thread::spawn(move || {
        unsafe { driver::ctx::set_current(std::ptr::null_mut()).unwrap() };
        assert!(driver::ctx::get_current().unwrap().is_none());
        start_signal.store(true, Ordering::Release);
        if explicit_clear {
            // Exercise the exposed cache accessor, not only the backend wrapper.
            backend.cuda_extension_cache().clear().unwrap();
            assert!(backend.cuda_extension_cache().is_empty().unwrap());
        } else {
            drop(backend);
        }
        assert!(
            driver::ctx::get_current().unwrap().is_none(),
            "restore caller context"
        );
    });
    while !started.load(Ordering::Acquire) {
        std::thread::yield_now();
    }
    std::thread::sleep(std::time::Duration::from_millis(100));
    release.store(true, Ordering::Release);
    retire.join().unwrap();
    assert_eq!(*result.lock().unwrap(), vec![(true, true, true)]);
    drop(runtime);
}

#[test]
#[ignore = "requires CUDA: real vendor handle and two worker streams"]
fn cuda_extension_cache_lifecycle_cross_thread_clear() {
    check_resource_retirement(true, false);
}

#[test]
#[ignore = "requires CUDA: real vendor handle and two worker streams"]
fn cuda_extension_cache_lifecycle_cross_thread_drop() {
    check_resource_retirement(false, false);
}

#[test]
#[ignore = "requires CUDA: real vendor handle and two worker streams"]
fn cuda_extension_cache_lifecycle_poisoned_drop_retires_resources() {
    check_resource_retirement(false, true);
}
