use super::{select_svd_driver, CusolverSvdRoutine, JAX_COMPATIBLE_GESVDJ_MAX_DIM};
use crate::extension::SvdDriver;

#[test]
fn auto_driver_keeps_jax_compatible_threshold() {
    let max = JAX_COMPATIBLE_GESVDJ_MAX_DIM;
    assert_eq!(max, 1024);
    for (m, n, expected) in [
        (1, 1, CusolverSvdRoutine::Gesvdj),
        (max, max, CusolverSvdRoutine::Gesvdj),
        (max + 1, 4, CusolverSvdRoutine::Gesvd),
        (4, max + 1, CusolverSvdRoutine::Gesvd),
        (max + 1, max + 1, CusolverSvdRoutine::Gesvd),
    ] {
        assert_eq!(
            select_svd_driver(SvdDriver::Auto, m, n),
            expected,
            "Auto policy for {m}x{n}"
        );
    }
}

#[test]
fn explicit_driver_overrides_dimension_policy() {
    let max = JAX_COMPATIBLE_GESVDJ_MAX_DIM;
    // Below the threshold Auto would pick gesvdj; Gesvd must still win.
    assert_eq!(
        select_svd_driver(SvdDriver::Gesvd, 64, 64),
        CusolverSvdRoutine::Gesvd
    );
    // Above the threshold Auto would pick gesvd; Gesvdj must still win.
    assert_eq!(
        select_svd_driver(SvdDriver::Gesvdj, max + 1, 4),
        CusolverSvdRoutine::Gesvdj
    );
    assert_eq!(
        select_svd_driver(SvdDriver::Gesvdj, 4, max + 1),
        CusolverSvdRoutine::Gesvdj
    );
}
