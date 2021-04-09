use crate::{
    core::{
        geometry::{Bounds2f, Bounds2i, Point2i},
        lowdiscrepancy::sobol_interval_to_index,
        pbrt::{is_power_of_2, log_2_int_i32, round_up_pow2_i32, round_up_pow2_i64},
        sampler::{GlobalSampler, Sampler},
    },
    impl_global_sampler,
};
use std::sync::Arc;

#[derive(Clone)]
pub struct SobolSampler {
    base: GlobalSampler,
    sample_bounds: Bounds2i,
    resolution: i32,
    log2_resolution: i32,
}

impl SobolSampler {
    pub fn new(mut samples_per_pixel: i64, sample_bounds: Bounds2i) -> Self {
        if !is_power_of_2(samples_per_pixel) {
            samples_per_pixel = round_up_pow2_i64(samples_per_pixel);
            log::warn!(
                "Non power-of-two sample count rounded up to {}",
                samples_per_pixel
            );
        }
        let resolution = round_up_pow2_i32(sample_bounds.diagonal().max_component());
        let log2_resolution = log_2_int_i32(resolution);
        Self {
            base: GlobalSampler::new(samples_per_pixel),
            sample_bounds,
            resolution,
            log2_resolution,
        }
    }
}

impl Sampler for SobolSampler {
    impl_global_sampler!();

    fn clone(&self, _seed: usize) -> Arc<Box<dyn Sampler>> {
        let ss = Clone::clone(self);
        Arc::new(Box::new(ss))
    }

    fn get_index_for_sample(&mut self, sample_num: usize) -> i64 {
        sobol_interval_to_index(
            self.log2_resolution as u32,
            sample_num as u64,
            &(self.current_pixel() - self.sample_bounds.min),
        ) as i64
    }

    fn sample_dimension(&self, _index: i64, _dimension: usize) -> f32 {
        todo!()
    }
}
