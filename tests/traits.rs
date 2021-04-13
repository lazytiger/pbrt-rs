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
