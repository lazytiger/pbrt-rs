use clap::Clap;
use pbrt::core::{pbrt_cleanup, pbrt_init, pbrt_parse_file};
use pbrt::Options;

fn main() {
    log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let opts = Options::parse();
    log::debug!("options:{:?}", opts);
    pbrt_init(&opts);
    for f in opts.scenes {
        pbrt_parse_file(f);
    }
    pbrt_cleanup();
}
