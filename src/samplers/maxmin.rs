use crate::core::{sampler::PixelSampler};

pub struct MaxMinDistSampler {
    base: PixelSampler,
    c_pixel: &'static [u32],
}

impl MaxMinDistSampler {
    pub fn new(_samples_per_pixel: i64, _n_sampled_dimensions: usize) -> Self {
        unimplemented!()
    }
}
