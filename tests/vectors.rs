use pbrt::core::RealNum;
use std::ops::Mul;

#[test]
fn test_vectors() {
    use pbrt::core::geometry::{Vector2f, Vector3f};

    let a = Vector2f::new(1.0, 2.0);
    assert_eq!(a[0], 1.0);
    assert_eq!(a[1], 2.0);

    let b = Vector3f::new(1.0, 2.0, 3.0);
    assert_eq!(b[0], 1.0);
    assert_eq!(b[1], 2.0);
    assert_eq!(b[2], 3.0);

    let a1 = Vector2f::new(1.0, 2.0);
    assert_eq!(a1, a);
    let a2 = Vector2f::new(2.0, 1.0);
    assert_ne!(a2, a);
    if a1 == a {}
}

#[test]
fn test_mul() {
    struct Vector {
        x: f32,
        y: f32,
    }

    impl Mul<f32> for Vector {
        type Output = Vector;

        fn mul(self, rhs: f32) -> Self::Output {
            Vector {
                x: self.x * rhs,
                y: self.y * rhs,
            }
        }
    }

    impl Mul for &Vector {
        type Output = Vector;

        fn mul(self, rhs: Self) -> Self::Output {
            Vector {
                x: self.x * rhs.x,
                y: self.y * rhs.y,
            }
        }
    }

    let mut a = Vector { x: 1.0, y: 2.0 };
    a = a * 2.0;

    a = &a * &a;
}

#[test]
fn from() {
    struct Point {
        x: f32,
        y: f32,
    }

    impl From<f32> for Point {
        fn from(p: f32) -> Self {
            println!("from invoked");
            Point { x: p, y: p }
        }
    }

    let mut a = Point { x: 1.0, y: 1.0 };
    a = Point::from(1.0f32);
}
fn inner_test<T: RealNum<T>>(a: T) -> T {
    a.sqrt()
}

#[test]
fn num() {
    //inner_test(1);
    inner_test(1.0);
}

#[test]
fn macro_test() {
    macro_rules! make_extent {
        ($o:ident, $($fields:ident),+) => {
            make_extent!($o, 0, $($fields,)+ [])
        };
        ($o:ident, $extent:expr, $field:ident, [$($after:ident),*]) => {
            if $($o.$field > $o.$after)&& * {
                $extent
            } else
            {make_extent!($o, $extent+1, [$($after,)* $field])}
        };
        ($o:ident, $extent:expr, $field:ident, $($before:ident,)+ [$($after:ident),*]) => {
            if $($o.$field > $o.$before) && * $(&& $o.$field > $o.$after)* {
                $extent
            } else
            {make_extent!($o, $extent+1, $($before,)* [$($after,)* $field])}
        };
        ($o:ident, $extent:expr, [$($fields:ident),+]) => {unreachable!()};
    }

    struct Vector3 {
        x: f32,
        y: f32,
        z: f32,
    }

    let a = Vector3 {
        x: 1.0,
        y: 2.0,
        z: 3.0,
    };

    let n = { make_extent!(a, x, y, z) };
    assert_eq!(n, 2);
}
