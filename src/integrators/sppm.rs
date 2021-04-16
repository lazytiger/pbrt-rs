use crate::{
    core::{
        camera::CameraDt,
        geometry::{Bounds2i, Bounds3f, Point2i, Point3f, Point3i, RayDifferentials, Vector3f},
        integrator::{compute_light_power_distribution, Integrator, SamplerIntegrator},
        parallel::parallel_for_2d,
        pbrt::{clamp, Float},
        reflection::BSDF,
        sampler::SamplerDtRw,
        scene::Scene,
        spectrum::Spectrum,
    },
    samplers::halton::HaltonSampler,
};
use derive_more::{Deref, DerefMut};
use std::{
    any::Any,
    sync::{atomic::AtomicUsize, Arc, RwLock},
};

pub struct SPPMIntegrator {
    camera: CameraDt,
    initial_search_radius: Float,
    n_iterations: usize,
    max_depth: usize,
    photon_per_iteration: usize,
    write_frequency: usize,
}

impl SPPMIntegrator {
    pub fn new(
        camera: CameraDt,
        n_iterations: usize,
        photon_per_iteration: isize,
        max_depth: usize,
        initial_search_radius: Float,
        write_frequency: usize,
    ) -> Self {
        let photon_per_iteration = if photon_per_iteration > 0 {
            photon_per_iteration as usize
        } else {
            camera.film().read().unwrap().cropped_pixel_bounds.area() as usize
        };
        Self {
            camera,
            n_iterations,
            photon_per_iteration,
            max_depth,
            initial_search_radius,
            write_frequency,
        }
    }
}

#[derive(Default, Clone)]
struct VisiblePoint {
    p: Point3f,
    wo: Vector3f,
    bsdf: Option<BSDF>,
    beta: Spectrum,
}

impl VisiblePoint {
    fn new(p: Point3f, wo: Vector3f, bsdf: Option<BSDF>, beta: Spectrum) -> Self {
        Self { p, wo, bsdf, beta }
    }
}

#[derive(Default, Clone)]
struct SPPMPixel {
    radius: Float,
    ld: Spectrum,
    vp: VisiblePoint,
    phi: Arc<RwLock<Vec<Float>>>,
    m: Arc<RwLock<usize>>,
    n: Float,
    tau: Spectrum,
}

impl SPPMPixel {
    fn new() -> Self {
        Self {
            radius: 0.0,
            ld: Default::default(),
            vp: Default::default(),
            phi: Arc::new(RwLock::new(vec![0.0; Spectrum::n_samples()])),
            m: Default::default(),
            n: 0.0,
            tau: Default::default(),
        }
    }
}

struct SPPMPixelListNode {
    pixel: SPPMPixel,
    next: Box<SPPMPixelListNode>,
}

fn to_grid(p: &Point3f, bounds: &Bounds3f, grid_res: &[i32; 3], pi: &mut Point3i) -> bool {
    let mut in_bounds = true;
    let pg = bounds.offset(p);
    for i in 0..3 {
        (*pi)[i] = (grid_res[i] as Float * pg[i]) as i32;
        in_bounds &= (*pi)[i] >= 0 && (*pi)[i] < grid_res[i];
        (*pi)[i] = clamp((*pi)[i], 0, grid_res[i] - 1);
    }
    in_bounds
}

fn hash(p: Point3i, hash_size: i32) -> usize {
    (((p.x * 73856093) ^ (p.y * 19349663) ^ (p.z * 83492791)) % hash_size) as usize
}

impl Integrator for SPPMIntegrator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn render(&mut self, scene: &Scene) {
        let pixel_bounds = self.camera.film().read().unwrap().cropped_pixel_bounds;
        let n_pixels = pixel_bounds.area();
        let mut pixels = vec![SPPMPixel::default(); n_pixels as usize];
        for i in 0..n_pixels as usize {
            pixels[i].radius = self.initial_search_radius;
        }
        let inv_sqrt_spp = 1.0 / (self.n_iterations as Float).sqrt();

        let light_distr = compute_light_power_distribution(scene);

        let sampler: SamplerDtRw = Arc::new(RwLock::new(Box::new(HaltonSampler::new(
            self.n_iterations as i64,
            &pixel_bounds,
            false,
        ))));

        let pixel_extent = pixel_bounds.diagonal();
        let tile_size = 16;
        let n_tiles = Point2i::new(
            (pixel_extent.x + tile_size - 1) / tile_size,
            (pixel_extent.y + tile_size - 1) / tile_size,
        );
        for iter in 0..self.n_iterations {
            parallel_for_2d(|tile| {}, &n_tiles);

            let mut grid_res = [0; 3];
            let mut grid_bounds = Bounds3f::default();
            let hash_size = n_pixels;
        }
    }
}
