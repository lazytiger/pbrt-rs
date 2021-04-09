use crate::{
    core::{
        geometry::Point2i,
        sampler::{PixelSampler, Sampler},
        sampling::{latin_hyper_cube, shuffle, stratified_sample_1d, stratified_sample_2d},
    },
    impl_pixel_sampler, inherit,
};
use std::sync::Arc;

#[derive(Clone)]
pub struct StratifiedSampler {
    base: PixelSampler,
    x_pixel_samples: usize,
    y_pixel_samples: usize,
    jitter_samples: bool,
}

impl StratifiedSampler {
    pub fn new(
        x_pixel_samples: usize,
        y_pixel_samples: usize,
        jitter_samples: bool,
        n_sampled_dimensions: usize,
    ) -> Self {
        Self {
            base: PixelSampler::new(
                (x_pixel_samples * y_pixel_samples) as i64,
                n_sampled_dimensions,
            ),
            x_pixel_samples,
            y_pixel_samples,
            jitter_samples,
        }
    }
}

inherit!(PixelSampler, StratifiedSampler, base);

impl Sampler for StratifiedSampler {
    impl_pixel_sampler!();

    fn start_pixel(&mut self, _p: Point2i) {
        for i in 0..self.samples_1d.len() {
            stratified_sample_1d(
                self.base.samples_1d[i].as_mut_slice(),
                self.x_pixel_samples * self.y_pixel_samples,
                &mut self.base.rng,
                self.jitter_samples,
            );
            shuffle(
                self.base.samples_1d[i].as_mut_slice(),
                self.x_pixel_samples * self.y_pixel_samples,
                1,
                &mut self.base.rng,
            );
        }

        for i in 0..self.samples_2d.len() {
            stratified_sample_2d(
                self.base.samples_2d[i].as_mut_slice(),
                self.x_pixel_samples,
                self.y_pixel_samples,
                &mut self.base.rng,
                self.jitter_samples,
            );
            shuffle(
                self.base.samples_2d[i].as_mut_slice(),
                self.x_pixel_samples * self.y_pixel_samples,
                1,
                &mut self.base.rng,
            );
        }

        for i in 0..self.samples_1d_array_sizes.len() {
            for j in 0..self.samples_per_pixel as usize {
                let count = self.samples_1d_array_sizes[i];
                stratified_sample_1d(
                    &mut self.base.base.sample_array_1d[i][j * count..],
                    count,
                    &mut self.base.rng,
                    self.jitter_samples,
                );
                shuffle(
                    &mut self.base.base.sample_array_1d[i][j * count..],
                    count,
                    1,
                    &mut self.base.rng,
                );
            }
        }

        for i in 0..self.samples_2d_array_sizes.len() {
            for j in 0..self.samples_per_pixel as usize {
                let count = self.samples_2d_array_sizes[i];
                latin_hyper_cube(
                    &mut self.base.base.sample_array_2d[i][j * count..],
                    count,
                    2,
                    &mut self.base.rng,
                );
            }
        }
    }

    fn clone(&self, seed: usize) -> Arc<Box<dyn Sampler>> {
        let mut ss = Clone::clone(self);
        ss.base.rng.set_sequence(seed);
        Arc::new(Box::new(ss))
    }
}
