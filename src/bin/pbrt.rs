use clap::Clap;
use pbrt::core::{pbrt_cleanup, pbrt_init, pbrt_parse_file};
use pbrt::make_vector;
use pbrt::Float;
use pbrt::Options;
use std::cmp::Eq;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

fn main() {
    log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let mut opts = Options::parse();
    if opts.threads == 0 {
        opts.threads = num_cpus::get();
    }
    log::debug!("options:{:?}", opts);
    let b = 1.0 / 0.0;
    log::debug!("nan:{}", f32::is_infinite(b));
    pbrt_init(&opts);
    for f in opts.scenes {
        pbrt_parse_file(f);
    }
    make_vector!(struct Vector4, x, y);
    pbrt_cleanup();
}

fn test(index: usize) -> usize {
    match index {
        x if x == 0 => 0,
        x if x == 0 + 1 => 1,
        _ => panic!("Hello"),
    }
}
