use crate::{
    core::{
        geometry::Point2i,
        lowdiscrepancy::{sobol_2d, van_der_corput},
        pbrt::round_up_pow2_i64,
        sampler::{PixelSampler, Sampler, SamplerDt},
    },
    impl_pixel_sampler,
};
use std::sync::Arc;

#[derive(Clone)]
pub struct ZeroTwoSequenceSampler {
    base: PixelSampler,
}

impl ZeroTwoSequenceSampler {
    pub fn new(samples_per_pixel: i64, n_sampled_dimension: usize) -> Self {
        Self {
            base: PixelSampler::new(round_up_pow2_i64(samples_per_pixel), n_sampled_dimension),
        }
    }
}

impl Sampler for ZeroTwoSequenceSampler {
    impl_pixel_sampler!();

    fn start_pixel(&mut self, p: Point2i) {
        for i in 0..self.base.samples_1d.len() {
            van_der_corput(
                1,
                self.base.samples_per_pixel as usize,
                &mut self.base.samples_1d[i][..],
                &mut self.base.rng,
            );
        }
        for i in 0..self.base.samples_2d.len() {
            sobol_2d(
                1,
                self.base.samples_per_pixel as usize,
                &mut self.base.samples_2d[i][..],
                &mut self.base.rng,
            );
        }
        for i in 0..self.base.samples_1d_array_sizes.len() {
            van_der_corput(
                self.base.samples_1d_array_sizes[i],
                self.base.samples_per_pixel as usize,
                &mut self.base.base.sample_array_1d[i][..],
                &mut self.base.rng,
            );
        }
        for i in 0..self.base.samples_2d_array_sizes.len() {
            sobol_2d(
                self.base.samples_2d_array_sizes[i],
                self.base.samples_per_pixel as usize,
                &mut self.base.base.sample_array_2d[i][..],
                &mut self.base.rng,
            );
        }
        self.base.start_pixel(p);
    }

    fn round_count(&self, n: usize) -> usize {
        round_up_pow2_i64(n as i64) as usize
    }

    fn clone(&self, seed: usize) -> SamplerDt {
        let mut zts = Clone::clone(self);
        zts.base.rng.set_sequence(seed);
        Arc::new(Box::new(zts))
    }
}
