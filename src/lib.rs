use clap::Clap;
use std::str::FromStr;

pub mod accelerators;
pub mod cameras;
pub mod core;
pub mod filters;
pub mod integrators;
pub mod lights;
pub mod materials;
pub mod media;
pub mod samplers;
#[macro_use]
pub mod shapes;
pub mod textures;

cfg_if::cfg_if! {
   if #[cfg(feature = "float64")] {
        pub type Float = f64;
        pub const PI: f64 = std::f64::consts::PI;
        #[repr(C)]
        pub(crate) union FloatUnion  {
            f:f64,
            u:u64,
        }
   } else {
        pub type Float = f32;
        pub const PI: f32 = std::f32::consts::PI;
        #[repr(C)]
        pub(crate) union FloatUnion {
            f:f32,
            u:u32,
        }
   }
}

#[derive(Debug)]
pub struct CropWindow {
    x0: Float,
    y0: Float,
    x1: Float,
    y1: Float,
}
impl FromStr for CropWindow {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let nums: Vec<Float> = s
            .split(" ")
            .map(|n| n.parse())
            .filter(|n| n.is_ok())
            .map(|n| n.unwrap())
            .collect();
        if nums.len() != 4 {
            Err("invalid crop window format")
        } else {
            Ok(CropWindow {
                x0: nums[0],
                y0: nums[1],
                x1: nums[2],
                y1: nums[3],
            })
        }
    }
}

#[derive(Clap, Debug)]
#[clap(
    version = "0.1",
    author = "Hoping White",
    about = "Another implementation based on pbrt using Rust language"
)]
pub struct Options {
    #[clap(
        short,
        long,
        default_value = "0",
        about = "Use specified number of threads for rendering."
    )]
    pub threads: usize,
    #[clap(
        short = 'r',
        long,
        about = "Automatically reduce a number of quality settings to render more quickly."
    )]
    pub quick_render: bool,
    #[clap(
        short,
        long,
        about = "Suppress all text output other than error messages."
    )]
    pub quiet: bool,
    #[clap(
        short,
        long,
        about = "Print a reformatted version of the input file(s) to standard output without rendering an image."
    )]
    pub cat: bool,
    #[clap(
        short = 'p',
        long,
        about = "Print a reformatted version of the input file(s) to a standard output and convert all triangle mesh to PLY files without rendering an image."
    )]
    pub to_ply: bool,
    #[clap(
        short,
        long,
        default_value = "pbrt.png",
        about = "Image file used for rendering output."
    )]
    pub image_file: String,
    #[clap(
        short = 'w',
        long,
        default_value = "0 1 0 1",
        about = "Specify an image window like this \"0.2 1.0 0.0 1.0\""
    )]
    pub crop_window: CropWindow,
    #[clap(about = "scene files used for rendering")]
    pub scenes: Vec<String>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            threads: 0,
            quick_render: false,
            quiet: false,
            cat: false,
            to_ply: false,
            image_file: "".to_string(),
            crop_window: CropWindow {
                x0: 0.0,
                y0: 1.0,
                x1: 0.0,
                y1: 1.0,
            },
            scenes: vec![],
        }
    }
}
