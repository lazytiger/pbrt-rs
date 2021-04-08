use crate::core::camera::CameraSample;
use crate::core::geometry::{Point2f, Point2i};
use crate::core::rng::RNG;
use crate::Float;
use std::any::Any;
use std::sync::Arc;

pub trait Sampler {
    fn as_any(&self) -> &dyn Any;
    fn start_pixel(&self, p: &Point2i);
    fn get_1d(&self) -> Float;
    fn get_2d(&self) -> Point2f;
    fn get_camera_sample(&self, p_raster: &Point2i) -> CameraSample {
        unimplemented!()
    }
    fn request_1d_array(&self, n: usize);
    fn request_2d_array(&self, n: usize);
    fn round_count(&self, n: usize) -> usize {
        n
    }
    fn get_1d_array(&self, n: usize) -> &[Float];
    fn get_2d_array(&self, n: usize) -> &[Point2f];
    fn start_next_sample(&self) -> bool;
    fn clone(&self, seed: usize) -> Arc<Box<dyn Sampler>>;
    fn set_sample_number(&self, sample_num: i64) -> bool;
    fn current_sample_number(&self) -> i64 {
        self.current_pixel_sample_index()
    }
    fn samples_per_pixel(&self) -> i64;
    fn current_pixel(&self) -> Point2i;
    fn current_pixel_sample_index(&self) -> i64;
    fn samples_1d_array_sizes(&self) -> &Vec<usize>;
    fn samples_2d_array_sizes(&self) -> &Vec<usize>;
    fn sample_array_1d(&self) -> Vec<Vec<Float>>;
    fn sample_array_2d(&self) -> Vec<Vec<Point2f>>;
    fn array_1d_offset(&self) -> usize;
    fn array_2d_offset(&self) -> usize;
}

pub(crate) struct BaseSampler {
    pub samples_per_pixel: i64,
    pub current_pixel: Point2i,
    pub current_pixel_sample_index: i64,
    pub samples_1d_array_sizes: Vec<usize>,
    pub samples_2d_array_sizes: Vec<usize>,
    pub sample_array_1d: Vec<Vec<Float>>,
    pub sample_array_2d: Vec<Vec<Point2f>>,
    array_1d_offset: usize,
    array_2d_offset: usize,
}

#[macro_export]
macro_rules! impl_base_sampler {
    () => {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn samples_per_pixel(&self) -> i64 {
            self.base.samples_per_pixel
        }
        fn current_pixel(&self) -> Point2i {
            self.base.current_pixel
        }
        fn current_pixel_sample_index(&self) -> i64 {
            self.base.current_pixel_sample_index
        }
        fn samples_1d_array_sizes(&self) -> &Vec<usize> {
            &self.base.samples_1d_array_sizes
        }
        fn samples_2d_array_sizes(&self) -> &Vec<usize> {
            &self.base.samples_2d_array_sizes
        }
        fn sample_array_1d(&self) -> Vec<Vec<Float>> {
            &self.base.sample_array_1d
        }
        fn sample_array_2d(&self) -> Vec<Vec<Point2f>> {
            &self.base.sample_array_2d
        }
        fn array_1d_offset(&self) -> usize {
            self.base.array_1d_offset
        }
        fn array_2d_offset(&self) -> usize {
            self.base.array_2d_offset
        }
    };
}

pub struct PixelSampler {
    base: BaseSampler,
    pub samples_1d: Vec<Vec<Float>>,
    pub samples_2d: Vec<Vec<Point2f>>,
    pub current_1d_dimension: usize,
    pub current_2d_dimension: usize,
    pub rng: RNG,
}
