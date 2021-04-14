use std::{any::Any, sync::Arc};

#[test]
fn test_traits() {
    trait Parent {
        fn as_any(&self) -> &dyn Any;
    }

    trait Phase {
        fn dummy(&self);
    }

    struct Test<T: Phase> {
        a: T,
    }

    impl<T: Phase + 'static> Parent for Test<T> {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    struct MyPhase {}

    impl Phase for MyPhase {
        fn dummy(&self) {
            println!("dummy called");
        }
    }

    let p: Arc<Box<dyn Parent>> = Arc::new(Box::new(Test { a: MyPhase {} }));
    let t: &Test<MyPhase> = p.as_any().downcast_ref().unwrap();
    t.a.dummy();
}

#[test]
fn test_two_level() {
    trait Level1 {
        fn as_any(&self) -> &dyn Any;
    }

    trait Level2: Level1 {
        fn test(&self);
    }

    struct Test {}

    impl Level1 for Test {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl Level2 for Test {
        fn test(&self) {
            println!("level2 called");
        }
    }

    let t: Arc<Box<dyn Level1>> = Arc::new(Box::new(Test {}));
    let l2: &Test = t.as_any().downcast_ref().unwrap();
    l2.test();
}
