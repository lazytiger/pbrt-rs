use crate::{
    core::{
        geometry::{Point2f, Point2i},
        lowdiscrepancy::{sample_generator_matrix, sobol_2d, van_der_corput, MAX_MIN_DIST},
        pbrt::{is_power_of_2, log_2_int_i64, round_up_pow2_i64},
        sampler::{PixelSampler, Sampler, SamplerDt},
        sampling::shuffle,
    },
    impl_pixel_sampler, Float,
};
use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

#[derive(Clone)]
pub struct MaxMinDistSampler {
    base: PixelSampler,
    c_pixel: &'static [u32],
}

impl MaxMinDistSampler {
    pub fn new(mut samples_per_pixel: i64, n_sampled_dimensions: usize) -> Self {
        let c_index = log_2_int_i64(samples_per_pixel);
        if c_index >= 17 {
            panic!("out of index");
        }
        if !is_power_of_2(samples_per_pixel) {
            samples_per_pixel = round_up_pow2_i64(samples_per_pixel);
        }
        Self {
            base: PixelSampler::new(samples_per_pixel, n_sampled_dimensions),
            c_pixel: &MAX_MIN_DIST[c_index as usize],
        }
    }

    pub fn round_count(&self, count: usize) -> usize {
        round_up_pow2_i64(count as i64) as usize
    }
}

impl Deref for MaxMinDistSampler {
    type Target = PixelSampler;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl DerefMut for MaxMinDistSampler {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

impl Sampler for MaxMinDistSampler {
    impl_pixel_sampler!();

    fn start_pixel(&mut self, p: Point2i) {
        let inv_spp = 1.0 / self.samples_per_pixel as Float;
        for i in 0..self.samples_per_pixel as usize {
            self.samples_2d[0][i] = Point2f::new(
                i as Float * inv_spp,
                sample_generator_matrix(self.c_pixel, i as u32, 0),
            );
        }
        let spp = self.samples_per_pixel as usize;
        shuffle(
            self.base.samples_2d[0].as_mut_slice(),
            spp,
            1,
            &mut self.base.rng,
        );
        for i in 0..self.samples_1d.len() {
            van_der_corput(
                1,
                spp,
                self.base.samples_1d[i].as_mut_slice(),
                &mut self.base.rng,
            );
        }

        for i in 0..self.samples_2d.len() {
            sobol_2d(
                1,
                spp,
                self.base.samples_2d[i].as_mut_slice(),
                &mut self.base.rng,
            );
        }

        for i in 0..self.samples_1d_array_sizes.len() {
            let count = self.samples_1d_array_sizes[i];
            van_der_corput(
                count,
                spp,
                self.base.base.sample_array_1d[i].as_mut_slice(),
                &mut self.base.rng,
            );
        }

        for i in 0..self.samples_2d_array_sizes.len() {
            let count = self.samples_2d_array_sizes[i];
            sobol_2d(
                count,
                spp,
                self.base.base.sample_array_2d[i].as_mut_slice(),
                &mut self.base.rng,
            );
        }
        self.base.start_pixel(p);
    }

    fn clone(&self, seed: usize) -> SamplerDt {
        let mut mmd = Clone::clone(self);
        mmd.rng.set_sequence(seed);
        Arc::new(Box::new(mmd))
    }
}
