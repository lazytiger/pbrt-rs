use pbrt::core::pbrt::{gamma, next_float_down, next_float_up};
use std::{intrinsics::transmute, marker::PhantomData, rc::Rc};

#[test]
fn test_float() {
    let suites = vec![-0.1 as f32, 0.1, -0.0, 0.0, 32.0, -32.0];
    for f in suites {
        println!("next_float_up({}) = {}", f, next_float_up(f));
        println!("next_float_down({}) = {}", f, next_float_down(f));
    }
}

#[test]
fn test_gama() {
    for f in &[0.1, 0.2, 10000.0] {
        println!("gama({}) = {}", *f, f * (1.0 + 2.0 * gamma(*f)));
    }
}

#[test]
fn test_transmute() {
    for f in &[0.1, 0.2, 1000.0] {
        let u: u64 = unsafe { transmute(*f) };
        let g: f64 = unsafe { transmute(u + 1) };
        println!("{} = {:02x}, {} = {:02x}", *f, u, g, u + 1);
    }
}

#[test]
fn test_lifetime() {
    struct Entity {}

    struct Container<'a> {
        root_index: usize,
        entities: &'a mut [Entity],
    }

    impl<'a> Container<'a> {
        fn root(&self) -> &Entity {
            &self.entities[self.root_index]
        }
    }
}

#[test]
fn test_for() {
    for mut i in 0..5 {
        println!("{}", i);
        if i == 3 {
            i -= 1;
        }
    }
}

#[test]
fn test_cast() {
    let a = 2;
    let b = &a;
    unsafe {
        let c = b as *const i32 as *mut i32;
        *c = 1;
    }
    println!("a = {}", a);
}

#[test]
fn expr_test() {
    let a = loop {
        break;
    };

    struct Test<const N: usize> {
        data: [f32; N],
    }
}

#[test]
fn local_test() {
    use std::{marker::PhantomData, sync::Once};

    #[derive(Copy, Clone)]
    pub struct MyFFI {
        call_fn: extern "C" fn() -> i32,
        phantom: PhantomData<*mut ()>, // !Send + !Sync
    }

    impl MyFFI {
        pub fn get() -> Option<Self> {
            static INIT: Once = Once::new();
            thread_local! {
                static FFI: Option<MyFFI> = {
                    let mut ffi = None;
                    INIT.call_once(|| ffi = Some(MyFFI::init()));
                    ffi
                }
            }

            FFI.with(|&ffi| ffi)
        }

        fn init() -> Self {
            todo!()
        }

        pub fn call(&self) -> i32 {
            (self.call_fn)()
        }
    }

    pub fn test() {
        std::thread::spawn(|| {
            let ffi = MyFFI::get().unwrap();
            ffi.call();
        });
    }
}
