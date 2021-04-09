use clap::Clap;
use pbrt::{
    core::{pbrt_cleanup, pbrt_init, pbrt_parse_file},
    Options,
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
    pbrt_cleanup();
}
