#[test]
fn test_vectors() {
    use pbrt::core::geometry::{Vector2f, Vector3f, Vector4f};

    let a = Vector2f::new(1.0, 2.0);
    assert_eq!(a[0], 1.0);
    assert_eq!(a[1], 2.0);

    let b = Vector3f::new(1.0, 2.0, 3.0);
    assert_eq!(b[0], 1.0);
    assert_eq!(b[1], 2.0);
    assert_eq!(b[2], 3.0);

    let c = Vector4f::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(c[0], 1.0);
    assert_eq!(c[1], 2.0);
    assert_eq!(c[2], 3.0);
    assert_eq!(c[3], 4.0);

    let a1 = Vector2f::new(1.0, 2.0);
    assert_eq!(a1, a);
    let a2 = Vector2f::new(2.0, 1.0);
    assert_ne!(a2, a);
    if a1 == a {}
}
