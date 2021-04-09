use crate::core::filter::Filter;
use crate::core::geometry::{Bounds2f, Bounds2i, Point2f, Point2i, Vector2f};
use crate::core::pbrt::Float;
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

impl Film {
    pub fn get_sample_bounds(&self) -> Bounds2i {
        let mut float_bounds = Bounds2f::new();
        float_bounds.min = Point2f::from(self.cropped_pixel_bounds.min).floor()
            + (Vector2f::new(0.5, 0.5) - *self.filter.radius());
        float_bounds.into()
    }

    pub fn get_physical_extent(&self) -> Bounds2f {
        unimplemented!()
    }
}
