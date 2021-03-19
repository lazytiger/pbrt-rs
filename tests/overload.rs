#[test]
fn test_overload() {
    trait Overload<T> {
        fn overload(&self, t: T);
    }

    struct Test {
        name: String,
    }

    impl Overload<&str> for Test {
        fn overload(&self, t: &str) {
            println!("overload({}) called", t);
        }
    }

    impl Overload<f32> for Test {
        fn overload(&self, t: f32) {
            println!("overload({}) called", t);
        }
    }

    impl Overload<(f32, &str)> for Test {
        fn overload(&self, t: (f32, &str)) {
            println!("overload({}, {}) called", t.0, t.1);
        }
    }

    let t = Test {
        name: "Hello".into(),
    };
    t.overload("world");
    t.overload(32.0);
    t.overload((32.0, "world"));
}
