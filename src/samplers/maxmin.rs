use crate::core::pbrt::log_2_int_i32;
use crate::core::sampler::PixelSampler;

pub struct MaxMinDistSampler {
    base: PixelSampler,
    c_pixel: &'static [u32],
}

impl MaxMinDistSampler {
    pub fn new(samples_per_pixel: i64, n_sampled_dimensions: usize) -> Self {
        unimplemented!()
    }
}
