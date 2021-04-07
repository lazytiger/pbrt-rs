use crate::core::filter::Filter;
use crate::core::geometry::{Bounds2i, Point2i};
use crate::Float;
use std::sync::Arc;

struct Pixel {
    xyz: [Float; 3],
    filter_weight_sum: Float,
    splat_xyz: [Float; 3],
    pad: Float,
}
const FILTER_TABLE_WIDTH: usize = 16;

pub struct Film {
    pub full_resolution: Point2i,
    pub diagonal: Float,
    pub filter: Arc<Box<dyn Filter>>,
    pub file_name: String,
    pub cropped_pixel_bounds: Bounds2i,
    pixels: Vec<Pixel>,
    filter_table: [Float; FILTER_TABLE_WIDTH * FILTER_TABLE_WIDTH],
    scale: Float,
    max_sample_luminance: Float,
}
