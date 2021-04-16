use crate::{
    core::{
        geometry::{Point2f, Point2i},
        rng::RNG,
        sampler::{BaseSampler, Sampler, SamplerDt, SamplerDtRw},
    },
    impl_base_sampler,
};
use std::sync::{Arc, RwLock};

#[derive(Clone)]
pub struct RandomSampler {
    base: BaseSampler,
    rng: RNG,
}

impl RandomSampler {
    pub fn new(ns: usize, seed: usize) -> Self {
        Self {
            base: BaseSampler::new(ns as i64),
            rng: RNG::new(seed),
        }
    }
}

impl Sampler for RandomSampler {
    impl_base_sampler!();

    fn start_pixel(&mut self, p: Point2i) {
        for i in 0..self.sample_array_1d().len() {
            for j in 0..self.sample_array_1d()[i].len() {
                self.sample_array_1d_mut()[i][j] = self.rng.uniform_float();
            }
        }
        for i in 0..self.sample_array_2d().len() {
            for j in 0..self.sample_array_2d()[i].len() {
                self.sample_array_2d_mut()[i][j] =
                    Point2f::new(self.rng.uniform_float(), self.rng.uniform_float());
            }
        }
        Sampler::start_pixel(self, p);
    }

    fn get_1d(&mut self) -> f32 {
        self.rng.uniform_float()
    }

    fn get_2d(&mut self) -> Point2f {
        Point2f::new(self.rng.uniform_float(), self.rng.uniform_float())
    }

    fn clone_sampler(&self, seed: usize) -> SamplerDtRw {
        let mut rs = Clone::clone(self);
        rs.rng.set_sequence(seed);
        Arc::new(RwLock::new(Box::new(rs)))
    }
}
