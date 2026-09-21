use super::*;
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::{EngineId, GraphCompiler, Runtime, TracedTensor};

fn complex_input(m: usize, n: usize, batches: usize) -> Tensor {
    tensor_c64(
        vec![m, n, batches],
        (0..m * n * batches)
            .map(|i| {
                let x = i as f64;
                Complex64::new(
                    (0.71 * x).sin() + if i % (m + 1) == 0 { 2.0 } else { 0.0 },
                    (0.37 * x + 0.2).cos(),
                )
            })
            .collect(),
    )
}

fn check_factors(input: &Tensor, factors: &[Tensor]) {
    let (m, n) = (input.shape()[0], input.shape()[1]);
    let k = m.min(n);
    let a = input.as_slice::<Complex64>().unwrap();
    let u = factors[0].as_slice::<Complex64>().unwrap();
    let s = factors[1].as_slice::<f64>().unwrap();
    let vt = factors[2].as_slice::<Complex64>().unwrap();
    assert_eq!(&factors[0].shape()[..2], &[m, k]);
    assert_eq!(&factors[2].shape()[..2], &[k, n]);
    for b in 0..a.len() / (m * n) {
        let u = &u[b * m * k..(b + 1) * m * k];
        let s = &s[b * k..(b + 1) * k];
        let vt = &vt[b * k * n..(b + 1) * k * n];
        let mut us = u.to_vec();
        for col in 0..k {
            for row in 0..m {
                us[row + m * col] *= s[col];
            }
        }
        let reconstructed = matmul_c64(&us, vt, m, k, n);
        assert_relative_error_c64(&reconstructed, &a[b * m * n..(b + 1) * m * n], 1e-12);
        for i in 0..k {
            for j in 0..k {
                let gram_u: Complex64 = (0..m).map(|r| u[r + m * i].conj() * u[r + m * j]).sum();
                let gram_v: Complex64 = (0..n).map(|c| vt[i + k * c] * vt[j + k * c].conj()).sum();
                let expected = Complex64::new(f64::from(i == j), 0.0);
                assert!(
                    (gram_u - expected).norm() < 1e-12,
                    "U orthogonality {i},{j}: {gram_u}"
                );
                assert!(
                    (gram_v - expected).norm() < 1e-12,
                    "V orthogonality {i},{j}: {gram_v}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA and cuSOLVER Xgesvdp"]
fn test_cubecl_svd_xgesvdp_rectangular_batched_and_values() {
    for (m, n) in [(7, 3), (3, 7), (5, 5)] {
        check_forced_svd_driver(m, n, SvdDriver::Xgesvdp);
        // Different matrices across eight outstanding calls stress host scratch
        // isolation as well as device offsets in the shared batch loop.
        let input = complex_input(m, n, 8);
        let mut gpu = gpu_backend();
        let device = upload(&gpu, &input);
        let options = SvdOptions::default().driver(SvdDriver::Xgesvdp);
        let outputs = with_cuda_linalg_session(&mut gpu, |s| {
            s.svd_with_options_read(TensorRead::from_tensor(&device), options)
        })
        .unwrap();
        let factors: Vec<_> = outputs.iter().map(|x| download(&gpu, x)).collect();
        check_factors(&input, &factors);
        let baseline = with_cuda_linalg_session(&mut gpu, |s| {
            s.svd_with_options(&device, SvdOptions::default().driver(SvdDriver::Gesvd))
        })
        .unwrap();
        assert_tensor_close(&factors[1], &download(&gpu, &baseline[1]), 1e-12);
        let values = with_cuda_linalg_session(&mut gpu, |s| {
            s.svd_values_with_driver_read(TensorRead::from_tensor(&device), SvdDriver::Xgesvdp)
        })
        .unwrap();
        assert_tensor_close(&download(&gpu, &values), &factors[1], 1e-12);
        assert_tensor_close(&download(&gpu, &device), &input, 0.0);
    }
}

#[test]
#[ignore = "requires CUDA and cuSOLVER Xgesvdp"]
fn test_cubecl_svd_xgesvdp_traced_and_pruned() {
    use tenferro_gpu::cuda::cuda_runtime_engine_registration;
    let input = complex_input(3, 7, 1);
    let mut gpu = gpu_backend();
    let device = upload(&gpu, &input);
    let options = SvdOptions::default().driver(SvdDriver::Xgesvdp);
    let expected =
        with_cuda_linalg_session(&mut gpu, |s| s.svd_with_options(&device, options)).unwrap();
    let traced = TracedTensor::from_tensor_concrete_shape(complex_input(3, 7, 1)).unwrap();
    let (u, s, vt) = traced.svd_with_options(options).unwrap();
    // A separate identity forces an S-only op instead of reusing the full SVD.
    let (_, only_s, _) = traced
        .svd_with_options(options.derivative_eps(2e-12))
        .unwrap();
    let (_, qr_s, _) = traced
        .svd_with_options(SvdOptions::default().driver(SvdDriver::Gesvd))
        .unwrap();
    let program = GraphCompiler::new()
        .compile_many(&[&u, &s, &vt, &only_s, &qr_s])
        .unwrap();
    let engine = EngineId::new("tenferro.test.cuda.xgesvdp.v1").unwrap();
    let mut builder = Runtime::builder();
    builder
        .register_engine(cuda_runtime_engine_registration(&gpu, engine.clone()).unwrap())
        .unwrap();
    builder
        .install_extension_module(tenferro_linalg::extension_module::<CudaBackend>(engine).unwrap())
        .unwrap();
    let outputs = builder
        .build()
        .unwrap()
        .run_compiled(&program, &[&device])
        .unwrap();
    let host: Vec<_> = outputs.iter().map(|x| download(&gpu, x)).collect();
    check_factors(&input, &host[..3]);
    for (actual, expected) in host[..3].iter().zip(&expected) {
        assert_tensor_close(actual, &download(&gpu, expected), 0.0);
    }
    assert_tensor_close(&host[3], &host[1], 1e-12);
    assert_tensor_close(&host[4], &host[1], 1e-12);
}

#[cfg(feature = "autodiff")]
#[test]
#[ignore = "requires CUDA and cuSOLVER Xgesvdp"]
fn test_cubecl_svd_xgesvdp_eager() {
    use tenferro_ad::{EagerRuntime, EagerTensor};
    use tenferro_linalg::EagerTensorLinalgExt;
    let input = complex_input(7, 3, 1);
    let mut gpu = gpu_backend();
    let device = upload(&gpu, &input);
    let options = SvdOptions::default().driver(SvdDriver::Xgesvdp);
    let expected =
        with_cuda_linalg_session(&mut gpu, |s| s.svd_with_options(&device, options)).unwrap();
    let runtime = EagerRuntime::with_cuda_backend(gpu.clone()).unwrap();
    let eager = EagerTensor::from_tensor_in(device, runtime).unwrap();
    let (u, s, vt) = eager.svd_with_options(options).unwrap();
    let factors: Vec<_> = [&u, &s, &vt]
        .into_iter()
        .map(|x| download(&gpu, &x.to_tensor().unwrap()))
        .collect();
    check_factors(&input, &factors);
    for (actual, expected) in factors.iter().zip(&expected) {
        assert_tensor_close(actual, &download(&gpu, expected), 0.0);
    }
}

#[test]
#[ignore = "requires CUDA; issue #1849 acceptance sizes"]
fn test_cubecl_svd_xgesvdp_large_spectrum_agrees_with_gesvd() {
    let mut gpu = gpu_backend();
    for m in [400, 800, 1024] {
        // A = F diag(s) Fᴴ times independent unit-modulus row/column phases.
        // The geometric sum constructs this dense ten-decade spectrum in O(m²).
        let ratio = 10f64.powf(-10.0 / (m - 1) as f64);
        let data = (0..m * m)
            .map(|idx| {
                let (i, j) = (idx % m, idx / m);
                let z = Complex64::from_polar(
                    ratio,
                    std::f64::consts::TAU * (i as f64 - j as f64) / m as f64,
                );
                Complex64::from_polar(
                    (1.0 - ratio.powi(m as i32)) / m as f64,
                    0.17 * i as f64 + 0.31 * j as f64,
                ) / (Complex64::new(1.0, 0.0) - z)
            })
            .collect();
        let input = tensor_c64(vec![m, m], data);
        let device = upload(&gpu, &input);
        let mut spectra = Vec::new();
        for driver in [SvdDriver::Gesvd, SvdDriver::Xgesvdp] {
            let result = with_cuda_linalg_session(&mut gpu, |s| {
                s.svd_with_options_read(
                    TensorRead::from_tensor(&device),
                    SvdOptions::default().driver(driver),
                )
            })
            .unwrap();
            spectra.push(
                download(&gpu, &result[1])
                    .as_slice::<f64>()
                    .unwrap()
                    .to_vec(),
            );
        }
        let error = spectra[0]
            .iter()
            .zip(&spectra[1])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
            / spectra[0][0];
        assert!(error < 1e-13, "{m}: max|Δs|/s₁ = {error}");
        for (i, &s) in spectra[1].iter().enumerate() {
            assert!(
                (s - ratio.powi(i as i32)).abs() < 1e-13,
                "{m}: known spectrum at {i}"
            );
        }
    }
}
