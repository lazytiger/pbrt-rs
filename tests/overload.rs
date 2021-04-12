#![feature(raw)]
use pbrt::core::pbrt::any_equal;
use std::any::Any;

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

#[test]
fn test_any() {
    trait Test {
        fn as_any(&self) -> &dyn Any;

        fn say_hello(&self);
    }

    struct MyTest {
        name: String,
    }

    struct HisTest {
        nick: String,
    }

    impl Test for MyTest {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn say_hello(&self) {
            println!("{} says hello", self.name);
        }
    }

    impl Test for HisTest {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn say_hello(&self) {
            println!("{} says hello", self.nick);
        }
    }

    let t1 = MyTest {
        name: "hoping".into(),
    };
    let t2 = HisTest {
        nick: "lazytiger".into(),
    };

    let t1: Box<dyn Test> = Box::new(t1);
    let t2: Box<dyn Test> = Box::new(t2);

    if any_equal(t1.as_any(), t2.as_any()) {
        println!("something wrong");
    }
    if any_equal(t1.as_any(), t1.as_any()) {
        println!("t1 passed");
    }
    if any_equal(t2.as_any(), t2.as_any()) {
        println!("t2 passed");
    }
}
