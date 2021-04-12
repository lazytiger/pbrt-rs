use pbrt::core::pbrt::{gamma, next_float_down, next_float_up};
use std::intrinsics::transmute;

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
