use crate::core::{
    camera::CameraSample,
    geometry::{Point2f, Point2i},
    pbrt::Float,
    rng::RNG,
};

use std::{
    any::Any,
    sync::{Arc, Mutex, RwLock},
};

use derive_more::{Deref, DerefMut};

pub trait Sampler {
    fn as_any(&self) -> &dyn Any;
    fn as_mut_any(&mut self) -> &mut dyn Any;
    fn start_pixel(&mut self, p: Point2i) {
        self.set_current_pixel(p);
        self.set_current_pixel_sample_index(0);
        self.set_array_1d_offset(0);
        self.set_array_2d_offset(0);
    }
    fn get_1d(&mut self) -> Float;
    fn get_2d(&mut self) -> Point2f;

    fn get_camera_sample(&mut self, p_raster: &Point2i) -> CameraSample {
        let mut cs = CameraSample::default();
        cs.p_film = Point2f::from(*p_raster) + self.get_2d();
        cs.time = self.get_1d();
        cs.p_lens = self.get_2d();
        cs
    }

    fn request_1d_array(&mut self, n: usize) {
        self.samples_2d_array_sizes_mut().push(n);
        let size = n * self.samples_per_pixel() as usize;
        self.sample_array_1d_mut().push(vec![0.0; size]);
    }

    fn request_2d_array(&mut self, n: usize) {
        self.samples_2d_array_sizes_mut().push(n);
        let size = n * self.samples_per_pixel() as usize;
        self.sample_array_2d_mut()
            .push(vec![Point2f::default(); size]);
    }

    fn round_count(&self, n: usize) -> usize {
        n
    }

    fn get_1d_array(&mut self, n: usize) -> Option<Vec<Float>> {
        if self.array_1d_offset() == self.sample_array_1d().len() {
            None
        } else {
            let offset = self.array_1d_offset();
            self.set_array_1d_offset(offset + 1);
            let array = self.sample_array_1d();
            let start = self.current_pixel_sample_index() as usize * n;
            Some(array[offset].as_slice()[start..].to_owned())
        }
    }
    fn get_2d_array(&mut self, n: usize) -> Option<Vec<Point2f>> {
        if self.array_2d_offset() == self.sample_array_2d().len() {
            None
        } else {
            let offset = self.array_2d_offset();
            self.set_array_2d_offset(offset + 1);
            let array = self.sample_array_2d();
            let start = self.current_pixel_sample_index() as usize * n;
            Some(array[offset].as_slice()[start..].to_owned())
        }
    }
    fn start_next_sample(&mut self) -> bool {
        self.set_array_1d_offset(0);
        self.set_array_2d_offset(0);
        self.set_current_pixel_sample_index(self.current_pixel_sample_index() + 1);
        self.current_pixel_sample_index() < self.samples_per_pixel()
    }
    fn clone_sampler(&self, seed: usize) -> SamplerDtRw;
    fn set_sample_number(&mut self, sample_num: i64) -> bool {
        self.set_array_1d_offset(0);
        self.set_array_2d_offset(0);
        self.set_current_pixel_sample_index(sample_num);
        self.current_pixel_sample_index() < self.samples_per_pixel()
    }
    fn current_sample_number(&self) -> i64 {
        self.current_pixel_sample_index()
    }
    fn get_index_for_sample(&mut self, sample_num: usize) -> i64 {
        unimplemented!("get_index_for_sample only implemented for GlobalSampler")
    }
    fn sample_dimension(&self, index: i64, dimension: usize) -> Float {
        unimplemented!("sample_dimension only implemented for GlobalSampler")
    }
    fn samples_per_pixel(&self) -> i64;

    #[inline]
    fn current_pixel(&self) -> Point2i;
    fn set_current_pixel(&mut self, p: Point2i);

    fn current_pixel_sample_index(&self) -> i64;
    fn set_current_pixel_sample_index(&mut self, index: i64);

    fn samples_1d_array_sizes(&self) -> &Vec<usize>;
    fn samples_1d_array_sizes_mut(&mut self) -> &mut Vec<usize>;

    fn samples_2d_array_sizes(&self) -> &Vec<usize>;
    fn samples_2d_array_sizes_mut(&mut self) -> &mut Vec<usize>;

    fn sample_array_1d(&self) -> &Vec<Vec<Float>>;
    fn sample_array_1d_mut(&mut self) -> &mut Vec<Vec<Float>>;

    fn sample_array_2d(&self) -> &Vec<Vec<Point2f>>;
    fn sample_array_2d_mut(&mut self) -> &mut Vec<Vec<Point2f>>;

    fn array_1d_offset(&self) -> usize;
    fn set_array_1d_offset(&mut self, offset: usize);

    fn array_2d_offset(&self) -> usize;
    fn set_array_2d_offset(&mut self, offset: usize);
}

#[derive(Clone)]
pub struct BaseSampler {
    pub samples_per_pixel: i64,
    pub current_pixel: Point2i,
    pub current_pixel_sample_index: i64,
    pub samples_1d_array_sizes: Vec<usize>,
    pub samples_2d_array_sizes: Vec<usize>,
    pub sample_array_1d: Vec<Vec<Float>>,
    pub sample_array_2d: Vec<Vec<Point2f>>,
    pub array_1d_offset: usize,
    pub array_2d_offset: usize,
}

impl BaseSampler {
    pub fn new(samples_per_pixel: i64) -> Self {
        Self {
            samples_per_pixel,
            current_pixel: Default::default(),
            current_pixel_sample_index: 0,
            samples_1d_array_sizes: vec![],
            samples_2d_array_sizes: vec![],
            sample_array_1d: vec![],
            sample_array_2d: vec![],
            array_1d_offset: 0,
            array_2d_offset: 0,
        }
    }
}

#[macro_export]
macro_rules! impl_base_sampler {
    () => {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
            self
        }

        #[inline]
        fn samples_per_pixel(&self) -> i64 {
            self.base.samples_per_pixel
        }

        #[inline]
        fn current_pixel(&self) -> $crate::core::geometry::Point2i {
            self.base.current_pixel
        }

        #[inline]
        fn set_current_pixel(&mut self, p: $crate::core::geometry::Point2i) {
            self.base.current_pixel = p;
        }

        #[inline]
        fn current_pixel_sample_index(&self) -> i64 {
            self.base.current_pixel_sample_index
        }

        #[inline]
        fn set_current_pixel_sample_index(&mut self, index: i64) {
            self.base.current_pixel_sample_index = index
        }

        #[inline]
        fn samples_1d_array_sizes(&self) -> &Vec<usize> {
            &self.base.samples_1d_array_sizes
        }

        #[inline]
        fn samples_1d_array_sizes_mut(&mut self) -> &mut Vec<usize> {
            &mut self.base.samples_1d_array_sizes
        }

        #[inline]
        fn samples_2d_array_sizes(&self) -> &Vec<usize> {
            &self.base.samples_2d_array_sizes
        }

        #[inline]
        fn samples_2d_array_sizes_mut(&mut self) -> &mut Vec<usize> {
            &mut self.base.samples_2d_array_sizes
        }

        #[inline]
        fn sample_array_1d(&self) -> &Vec<Vec<$crate::core::pbrt::Float>> {
            &self.base.sample_array_1d
        }

        #[inline]
        fn sample_array_1d_mut(&mut self) -> &mut Vec<Vec<$crate::core::pbrt::Float>> {
            &mut self.base.sample_array_1d
        }

        #[inline]
        fn sample_array_2d(&self) -> &Vec<Vec<$crate::core::geometry::Point2f>> {
            &self.base.sample_array_2d
        }

        #[inline]
        fn sample_array_2d_mut(&mut self) -> &mut Vec<Vec<$crate::core::geometry::Point2f>> {
            &mut self.base.sample_array_2d
        }

        #[inline]
        fn array_1d_offset(&self) -> usize {
            self.base.array_1d_offset
        }

        #[inline]
        fn set_array_1d_offset(&mut self, offset: usize) {
            self.base.array_1d_offset = offset;
        }

        #[inline]
        fn array_2d_offset(&self) -> usize {
            self.base.array_2d_offset
        }

        #[inline]
        fn set_array_2d_offset(&mut self, offset: usize) {
            self.base.array_2d_offset = offset;
        }
    };
}

#[derive(Clone, Deref, DerefMut)]
pub struct PixelSampler {
    #[deref]
    #[deref_mut]
    pub(crate) base: BaseSampler,
    pub samples_1d: Vec<Vec<Float>>,
    pub samples_2d: Vec<Vec<Point2f>>,
    pub current_1d_dimension: usize,
    pub current_2d_dimension: usize,
    pub rng: RNG,
}

impl PixelSampler {
    pub fn new(sample_per_pixel: i64, n_sample_dimensions: usize) -> Self {
        let mut samples_1d = Vec::with_capacity(n_sample_dimensions);
        let mut samples_2d = Vec::with_capacity(n_sample_dimensions);
        for _i in 0..n_sample_dimensions {
            samples_1d.push(vec![0.0; sample_per_pixel as usize]);
            samples_2d.push(vec![Point2f::default(); sample_per_pixel as usize]);
        }
        Self {
            base: BaseSampler::new(sample_per_pixel),
            samples_1d,
            samples_2d,
            current_1d_dimension: 0,
            current_2d_dimension: 0,
            rng: Default::default(),
        }
    }
}

impl Sampler for PixelSampler {
    impl_base_sampler!();

    fn get_1d(&mut self) -> Float {
        let current = self.current_1d_dimension;
        if current < self.samples_1d.len() {
            self.current_1d_dimension += 1;
            self.samples_1d[current][self.current_pixel_sample_index as usize]
        } else {
            self.rng.uniform_float()
        }
    }

    fn get_2d(&mut self) -> Point2f {
        let current = self.current_2d_dimension;
        if current < self.samples_2d.len() {
            self.current_2d_dimension += 1;
            self.samples_2d[current][self.current_pixel_sample_index as usize]
        } else {
            Point2f::new(self.rng.uniform_float(), self.rng.uniform_float())
        }
    }

    fn start_next_sample(&mut self) -> bool {
        self.current_1d_dimension = 0;
        self.current_2d_dimension = 0;
        Sampler::start_next_sample(self)
    }

    fn clone_sampler(&self, _seed: usize) -> SamplerDtRw {
        unimplemented!()
    }

    fn set_sample_number(&mut self, sample_num: i64) -> bool {
        self.current_2d_dimension = 0;
        self.current_1d_dimension = 0;
        Sampler::set_sample_number(self, sample_num)
    }
}

const ARRAY_START_DIM: usize = 5;

#[derive(Clone, Deref, DerefMut)]
pub struct GlobalSampler {
    #[deref]
    #[deref_mut]
    base: BaseSampler,
    dimension: usize,
    interval_sample_index: i64,
    array_end_dim: usize,
}

impl GlobalSampler {
    pub fn new(samples_per_pixel: i64) -> Self {
        Self {
            base: BaseSampler::new(samples_per_pixel),
            dimension: 0,
            interval_sample_index: 0,
            array_end_dim: 0,
        }
    }
}

impl Sampler for GlobalSampler {
    impl_base_sampler!();

    fn start_pixel(&mut self, _p: Point2i) {
        self.dimension = 0;
        self.interval_sample_index = self.get_index_for_sample(0);
        self.array_end_dim =
            ARRAY_START_DIM + self.sample_array_1d.len() + 2 * self.sample_array_2d.len();
        for i in 0..self.samples_1d_array_sizes.len() {
            let n_samples = self.samples_1d_array_sizes[i] * self.samples_per_pixel as usize;
            for j in 0..n_samples {
                let index = self.get_index_for_sample(j);
                self.sample_array_1d[i][j] = self.sample_dimension(index, ARRAY_START_DIM + i);
            }
        }

        let mut dim = ARRAY_START_DIM + self.samples_1d_array_sizes.len();
        for i in 0..self.samples_2d_array_sizes.len() {
            let n_samples = self.samples_2d_array_sizes[i] * self.samples_per_pixel as usize;
            for j in 0..n_samples {
                let idx = self.get_index_for_sample(j);
                self.sample_array_2d[i][j].x = self.sample_dimension(idx, dim);
                self.sample_array_2d[i][j].y = self.sample_dimension(idx, dim + 1);
            }
            dim += 2;
        }
    }

    fn get_1d(&mut self) -> Float {
        if self.dimension >= ARRAY_START_DIM && self.dimension < self.array_end_dim {
            self.dimension = self.array_end_dim;
        }
        let f = self.sample_dimension(self.interval_sample_index, self.dimension);
        self.dimension += 1;
        f
    }

    fn get_2d(&mut self) -> Point2f {
        if self.dimension + 1 >= ARRAY_START_DIM && self.dimension < self.array_end_dim {
            self.dimension = self.array_end_dim;
        }
        let p = Point2f::new(
            self.sample_dimension(self.interval_sample_index, self.dimension),
            self.sample_dimension(self.interval_sample_index, self.dimension + 1),
        );
        self.dimension += 2;
        p
    }

    fn start_next_sample(&mut self) -> bool {
        self.dimension = 0;
        self.interval_sample_index =
            self.get_index_for_sample(self.current_pixel_sample_index as usize + 1);
        Sampler::start_next_sample(self)
    }

    fn clone_sampler(&self, _seed: usize) -> SamplerDtRw {
        unimplemented!()
    }

    fn set_sample_number(&mut self, sample_num: i64) -> bool {
        self.dimension = 0;
        self.interval_sample_index = self.get_index_for_sample(sample_num as usize);
        Sampler::set_sample_number(self, sample_num)
    }
}

#[macro_export]
macro_rules! impl_global_sampler {
    () => {
        crate::impl_base_sampler!();

        fn get_1d(&mut self) -> $crate::core::pbrt::Float {
            self.base.get_1d()
        }

        fn get_2d(&mut self) -> $crate::core::geometry::Point2f {
            self.base.get_2d()
        }
    };
}

#[macro_export]
macro_rules! impl_pixel_sampler {
    () => {
        crate::impl_base_sampler!();

        fn get_1d(&mut self) -> $crate::core::pbrt::Float {
            self.base.get_1d()
        }

        fn get_2d(&mut self) -> $crate::core::geometry::Point2f {
            self.base.get_2d()
        }

        fn start_next_sample(&mut self) -> bool {
            self.base.start_next_sample()
        }

        fn set_sample_number(&mut self, sample_num: i64) -> bool {
            self.set_sample_number(sample_num)
        }
    };
}

pub type SamplerDt = Arc<Box<dyn Sampler + Send + Sync>>;
pub type SamplerDtMut = Arc<Mutex<Box<dyn Sampler + Send + Sync>>>;
pub type SamplerDtRw = Arc<RwLock<Box<dyn Sampler + Send + Sync>>>;
