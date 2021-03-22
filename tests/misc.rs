use pbrt::core::{next_float_down, next_float_up};

#[test]
fn test_float() {
    let suites = vec![-0.1 as f32, 0.1, -0.0, 0.0, 32.0, -32.0];
    for f in suites {
        println!("next_float_up({}) = {}", f, next_float_up(f));
        println!("next_float_down({}) = {}", f, next_float_down(f));
    }
}
