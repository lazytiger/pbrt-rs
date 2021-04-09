use crate::{
    core::{
        geometry::{Point2f, Point2i},
        rng::RNG,
        sampler::{BaseSampler, Sampler},
    },
    impl_base_sampler,
};
use std::sync::Arc;

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

    fn get_1d(&mut self) -> f32 {
        self.rng.uniform_float()
    }

    fn get_2d(&mut self) -> Point2f {
        Point2f::new(self.rng.uniform_float(), self.rng.uniform_float())
    }

    fn clone(&self, seed: usize) -> Arc<Box<dyn Sampler>> {
        let mut rs = Clone::clone(self);
        rs.rng.set_sequence(seed);
        Arc::new(Box::new(rs))
    }

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

    fn get_index_for_sample(&mut self, _sample_num: usize) -> i64 {
        unimplemented!("This method is not supported for RandomSampler")
    }

    fn sample_dimension(&self, _index: i64, _dimension: usize) -> f32 {
        unimplemented!("This method is not supported for RandomSampler")
    }
}
