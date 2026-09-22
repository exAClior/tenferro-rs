use super::shared_workspace_capacity;

#[test]
fn shared_workspace_capacity_handles_zero_floor_growth_and_overflow() {
    for (request, expected) in [
        (0, 0),
        (1, 1 << 20),
        ((1 << 20) - 1, 1 << 20),
        (1 << 20, 1 << 20),
        ((1 << 20) + 1, 1 << 21),
        (3 << 20, 1 << 22),
        (1 << 63, 1 << 63),
    ] {
        assert_eq!(shared_workspace_capacity(request).unwrap(), expected);
    }
    assert!(shared_workspace_capacity((1 << 63) + 1).is_err());
    assert!(shared_workspace_capacity(u64::MAX).is_err());
}
