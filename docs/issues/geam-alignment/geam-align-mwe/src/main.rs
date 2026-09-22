//! cuBLAS host-pointer-mode coefficients for cublasZgeam must be 16-byte
//! aligned (`cuDoubleComplex` is `double2`, declared align(16)). A
//! `num_complex::Complex64` is only 8-byte aligned, so `&Complex64 as *const
//! cuDoubleComplex` is a valid pointer only when the stack happens to place the
//! value at a multiple of 16. This program removes the luck: it stores alpha and
//! beta inside a 16-byte aligned struct at a chosen byte offset and calls
//! cublasZgeam exactly like tenferro-gpu's `geam_accum` (y <- alpha*x + beta*y,
//! n x 1 column-major matrices, B == C).
//!
//!   cargo run --release -- 0    # alpha at offset 0 (16-byte aligned): works
//!   cargo run --release -- 8    # alpha at offset 8 (8 mod 16): SIGSEGV
use cudarc::cublas::{sys, CudaBlas};
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use num_complex::Complex64;

// 16-byte aligned container; the 8-byte `pad` puts `alpha` at offset 8 and
// `beta` at offset 24, i.e. both at 8 mod 16, exactly what a `Complex64`
// local is allowed to get on the stack.
#[repr(C, align(16))]
struct Coefficients {
    pad: f64,
    alpha: Complex64,
    beta: Complex64,
}

fn main() {
    let offset: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    assert!(offset == 0 || offset == 8, "offset must be 0 or 8");
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).expect("cuBLAS handle");
    blas.set_pointer_mode(sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST).unwrap();

    let n = 4usize;
    // x = [1+2i, ...], y = [10+20i, ...] stored as (re, im) f64 pairs.
    let x_host: Vec<f64> = (0..n).flat_map(|i| [i as f64 + 1.0, 2.0 * (i as f64 + 1.0)]).collect();
    let y_host: Vec<f64> = (0..n).flat_map(|i| [10.0 * (i as f64 + 1.0), 20.0 * (i as f64 + 1.0)]).collect();
    let x_dev = stream.clone_htod(&x_host).unwrap();
    let mut y_dev = stream.clone_htod(&y_host).unwrap();

    let c = Coefficients { pad: 0.0, alpha: Complex64::new(2.0, 0.0), beta: Complex64::new(1.0, 0.0) };
    // offset 0: 16-byte aligned copies; offset 8: the fields inside `c`.
    #[repr(C, align(16))]
    struct Aligned(Complex64);
    let (aligned_alpha, aligned_beta) = (Aligned(c.alpha), Aligned(c.beta));
    let (alpha_ptr, beta_ptr): (*const Complex64, *const Complex64) = if offset == 0 {
        (&aligned_alpha.0, &aligned_beta.0)
    } else {
        (&c.alpha, &c.beta)
    };
    println!("alpha at {:p} (mod 16 = {}), beta at {:p} (mod 16 = {})",
             alpha_ptr, alpha_ptr as usize % 16, beta_ptr, beta_ptr as usize % 16);
    assert_eq!(alpha_ptr as usize % 16, offset % 16, "container layout did not give the requested offset");

    let ld = n as i32;
    let status = {
        let (x_ptr, _gx) = x_dev.device_ptr(&stream);
        let (y_ptr, _gy) = y_dev.device_ptr_mut(&stream);
        unsafe {
        sys::cublasZgeam(
            *blas.handle(),
            sys::cublasOperation_t::CUBLAS_OP_N,
            sys::cublasOperation_t::CUBLAS_OP_N,
            n as i32,
            1,
            alpha_ptr.cast::<sys::cuDoubleComplex>(),   // <- same cast as tenferro-gpu blas1.rs
            x_ptr as *const sys::cuDoubleComplex,
            ld,
            beta_ptr.cast::<sys::cuDoubleComplex>(),
            y_ptr as *const sys::cuDoubleComplex,
            ld,
            y_ptr as *mut sys::cuDoubleComplex,
            ld,
        )
        }
    };
    stream.synchronize().unwrap();
    let y_out = stream.clone_dtoh(&y_dev).unwrap();
    println!("status = {:?}", status);
    println!("y = 2*x + y = {:?}", y_out);
    let expected: Vec<f64> = x_host.iter().zip(&y_host).map(|(x, y)| 2.0 * x + y).collect();
    assert_eq!(y_out, expected);
    println!("OK");
}
