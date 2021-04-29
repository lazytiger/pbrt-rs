use crate::{
    core::{
        camera::CameraDt,
        geometry::{
            Bounds2i, Bounds3f, Normal3f, Point2f, Point2i, Point3f, Point3i, RayDifferentials,
            Union, Vector3f,
        },
        integrator::{
            compute_light_power_distribution, uniform_sample_one_light, Integrator,
            SamplerIntegrator,
        },
        interaction::SurfaceInteraction,
        lowdiscrepancy::radical_inverse,
        material::TransportMode,
        memory::{Arena, Indexed},
        pbrt::{clamp, lerp, Float},
        reflection::{BxDFType, BSDF},
        sampler::SamplerDtRw,
        scene::Scene,
        spectrum::Spectrum,
    },
    parallel_for, parallel_for_2d,
    samplers::halton::HaltonSampler,
};
use derive_more::{Deref, DerefMut};
use std::{
    any::Any,
    f32::consts::PI,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, RwLock,
    },
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
    bsdf: Option<Arc<BSDF>>,
    beta: Spectrum,
}

impl VisiblePoint {
    fn new(p: Point3f, wo: Vector3f, bsdf: Option<Arc<BSDF>>, beta: Spectrum) -> Self {
        Self { p, wo, bsdf, beta }
    }
}

#[derive(Default, Clone)]
struct SPPMPixel {
    radius: Float,
    ld: Spectrum,
    vp: VisiblePoint,
    phi: Vec<Float>,
    m: usize,
    n: Float,
    tau: Spectrum,
}

impl SPPMPixel {
    fn new() -> Self {
        Self {
            radius: 0.0,
            ld: Default::default(),
            vp: Default::default(),
            phi: vec![0.0; Spectrum::n_samples()],
            m: Default::default(),
            n: 0.0,
            tau: Default::default(),
        }
    }
}

#[derive(Default, Copy, Clone)]
struct SPPMPixelListNode {
    index: usize,
    pixel: usize,
    next: Option<usize>,
}

impl Indexed for SPPMPixelListNode {
    fn index(&self) -> usize {
        self.index
    }

    fn set_index(&mut self, index: usize) {
        self.index = index;
    }
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
        let mut pixels = Vec::new();
        pixels.resize_with(n_pixels as usize, || {
            let mut pixel = SPPMPixel::default();
            pixel.radius = self.initial_search_radius;
            Arc::new(pixel)
        });
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
        let arena = RwLock::new(Arena::with_capacity(1024));
        for iter in 0..self.n_iterations {
            parallel_for_2d!(
                |tile: Point2i| {
                    let tile_index = tile.y * n_tiles.x + tile.x;
                    let tile_sampler = sampler.read().unwrap().clone_sampler(tile_index as usize);
                    let x0 = pixel_bounds.min.x + tile.x * tile_size;
                    let x1 = std::cmp::min(x0 + tile_size, pixel_bounds.max.x);
                    let y0 = pixel_bounds.min.y + tile.y * tile_size;
                    let y1 = std::cmp::min(y0 + tile_size, pixel_bounds.max.y);
                    let tile_bounds = Bounds2i::from((Point2i::new(x0, y0), Point2i::new(x1, y1)));
                    for p_pixel in &tile_bounds {
                        let camera_sampler = {
                            let mut lock = tile_sampler.write().unwrap();
                            lock.start_pixel(p_pixel);
                            lock.set_sample_number(iter as i64);
                            lock.get_camera_sample(&p_pixel)
                        };
                        let mut ray = RayDifferentials::default();
                        let mut beta: Spectrum = self
                            .camera
                            .generate_ray_differential(&camera_sampler, &mut ray)
                            .into();
                        if beta.is_black() {
                            continue;
                        }
                        ray.scale_differentials(inv_sqrt_spp);

                        let p_pixel_o = p_pixel - pixel_bounds.min;
                        let pixel_offset =
                            p_pixel_o.x + p_pixel_o.y * (pixel_bounds.max.x - pixel_bounds.min.x);
                        let mut pixel = pixels[pixel_offset as usize].clone();
                        let pixel_mut = Arc::get_mut(&mut pixel).unwrap();
                        let mut specular_bounce = false;
                        let mut depth = 0;
                        while depth < self.max_depth {
                            let mut isect = SurfaceInteraction::default();
                            if !scene.intersect(&mut ray.base, &mut isect) {
                                let pixel = pixels[pixel_offset as usize].clone();
                                for light in &scene.lights {
                                    pixel_mut.ld += beta * light.le(&ray);
                                }
                                break;
                            }
                            isect.compute_scattering_functions(&ray, true, TransportMode::Radiance);
                            if isect.bsdf.is_none() {
                                ray = isect.spawn_ray(&ray.d).into();
                                depth -= 1;
                                continue;
                            }

                            let bsdf = isect.bsdf.as_ref().unwrap();
                            let wo = -ray.d;
                            if depth == 0 || specular_bounce {
                                pixel_mut.ld += beta * isect.le(&wo);
                            }
                            pixel_mut.ld += beta
                                * uniform_sample_one_light(
                                    &isect,
                                    scene,
                                    tile_sampler.clone(),
                                    false,
                                    None,
                                );
                            let is_diffuse = bsdf.num_components(
                                BxDFType::BSDF_DIFFUSE
                                    | BxDFType::BSDF_REFLECTION
                                    | BxDFType::BSDF_TRANSMISSION,
                            ) > 0;
                            let is_glossy = bsdf.num_components(
                                BxDFType::BSDF_GLOSSY
                                    | BxDFType::BSDF_REFLECTION
                                    | BxDFType::BSDF_TRANSMISSION,
                            ) > 0;
                            if is_diffuse || (is_glossy && depth == self.max_depth - 1) {
                                pixel_mut.vp =
                                    VisiblePoint::new(isect.p, wo, Some(bsdf.clone()), beta);
                                break;
                            }

                            if depth < self.max_depth - 1 {
                                let mut pdf = 0.0;
                                let mut wi = Vector3f::default();
                                let mut typ = BxDFType::empty();
                                let f = bsdf.sample_f(
                                    &wo,
                                    &mut wi,
                                    &tile_sampler.write().unwrap().get_2d(),
                                    &mut pdf,
                                    BxDFType::all(),
                                    Some(&mut typ),
                                );
                                if pdf == 0.0 || f.is_black() {
                                    break;
                                }
                                specular_bounce = typ.contains(BxDFType::BSDF_SPECULAR);
                                beta *= f * wi.abs_dot(&isect.shading.n) / pdf;
                                if beta.y_value() < 0.25 {
                                    let continue_prob = beta.y_value().min(1.0);
                                    if tile_sampler.write().unwrap().get_1d() > continue_prob {
                                        break;
                                    }
                                    beta /= continue_prob;
                                }
                                ray = isect.spawn_ray(&wi).into();
                            }
                            depth += 1;
                        }
                    }
                },
                n_tiles
            );

            let mut grid_res = [0; 3];
            let mut grid_bounds = Bounds3f::default();
            let hash_size = n_pixels;
            let mut grid = Vec::new();
            grid.resize_with(hash_size as usize, || AtomicUsize::new(0));
            let mut max_radius = 0.0 as Float;
            {
                for i in 0..n_pixels as usize {
                    let pixel = pixels[i].clone();
                    if pixel.vp.beta.is_black() {
                        continue;
                    }
                    let vp_bound = Bounds3f::from(pixel.vp.p).expand(pixel.radius);
                    grid_bounds = grid_bounds.union(&vp_bound);
                    max_radius = max_radius.max(pixel.radius);
                }
            }

            let diag = grid_bounds.diagonal();
            let max_diag = diag.max_component();
            let base_grid_res = (max_diag / max_radius) as usize;
            for i in 0..3 {
                grid_res[i] = (base_grid_res as Float * diag[i] / max_diag).max(1.0) as i32;
            }

            parallel_for!(
                |pixel_index: usize| {
                    let pixel = pixels[pixel_index].clone();
                    if pixel.vp.beta.is_black() {
                        let radius = pixel.radius;
                        let mut p_min = Point3i::default();
                        let mut p_max = Point3i::default();
                        to_grid(
                            &(pixel.vp.p - Vector3f::new(radius, radius, radius)),
                            &grid_bounds,
                            &grid_res,
                            &mut p_min,
                        );
                        to_grid(
                            &(pixel.vp.p + Vector3f::new(radius, radius, radius)),
                            &grid_bounds,
                            &grid_res,
                            &mut p_max,
                        );

                        for z in p_min.z..p_max.z + 1 {
                            for y in p_min.y..p_max.y + 1 {
                                for x in p_min.x..p_max.x + 1 {
                                    let h = hash(Point3i::new(x, y, z), hash_size);
                                    let node = {
                                        let mut lock = arena.write().unwrap();
                                        let (index, node) =
                                            lock.alloc(SPPMPixelListNode::default());
                                        node.pixel = pixel_index;
                                        node.next = Some(grid[h].load(Ordering::SeqCst));
                                        node.clone()
                                    };
                                    while let Err(_) = grid[h].compare_exchange_weak(
                                        node.next.unwrap(),
                                        node.index,
                                        Ordering::SeqCst,
                                        Ordering::SeqCst,
                                    ) {}
                                }
                            }
                        }
                    }
                },
                n_pixels as usize,
                4096
            );

            parallel_for!(
                |photon_index: usize| {
                    let halton_index = (iter * self.photon_per_iteration + photon_index) as u64;
                    let mut halton_dim = 0;
                    let mut light_pdf = 0.0;
                    let light_sample = radical_inverse(halton_dim, halton_index);
                    halton_dim += 1;
                    let light_num = light_distr.as_ref().unwrap().sample_discrete(
                        light_sample,
                        Some(&mut light_pdf),
                        None,
                    );
                    let light = scene.lights[light_num].clone();

                    let u_light0 = Point2f::new(
                        radical_inverse(halton_dim, halton_index),
                        radical_inverse(halton_dim + 1, halton_index),
                    );
                    let u_light1 = Point2f::new(
                        radical_inverse(halton_dim + 2, halton_index),
                        radical_inverse(halton_dim + 3, halton_index),
                    );
                    let u_light_time = lerp(
                        radical_inverse(halton_dim + 4, halton_index),
                        self.camera.shutter_open(),
                        self.camera.shutter_close(),
                    );
                    halton_dim += 5;

                    let mut photon_ray = RayDifferentials::default();
                    let mut n_light = Normal3f::default();
                    let (mut pdf_pos, mut pdf_dir) = (0.0, 0.0);
                    let le = light.sample_le(
                        &u_light0,
                        &u_light1,
                        u_light_time,
                        &mut photon_ray,
                        &mut n_light,
                        &mut pdf_pos,
                        &mut pdf_dir,
                    );
                    if pdf_pos == 0.0 || pdf_dir == 0.0 || le.is_black() {
                        return;
                    }

                    let mut beta =
                        le * n_light.abs_dot(&photon_ray.d) / (light_pdf * pdf_pos * pdf_dir);
                    if beta.is_black() {
                        return;
                    }

                    let mut isect = SurfaceInteraction::default();
                    let mut depth = 0;
                    while depth < self.max_depth {
                        if !scene.intersect(&mut photon_ray.base, &mut isect) {
                            break;
                        }
                        if depth > 0 {
                            let mut photon_grid_index = Point3i::default();
                            if to_grid(&isect.p, &grid_bounds, &grid_res, &mut photon_grid_index) {
                                let h = hash(photon_grid_index, hash_size);

                                let mut node_index = grid[h].load(Ordering::Relaxed);
                                while let Some(node) = arena.read().unwrap().get(node_index) {
                                    let mut pixel = pixels[node.pixel].clone();
                                    let pixel_mut = Arc::get_mut(&mut pixel).unwrap();
                                    let radius = pixel_mut.radius;
                                    if pixel_mut.vp.p.distance_square(&isect.p) > radius * radius {
                                        continue;
                                    }
                                    let mut wi = -photon_ray.d;
                                    let phi = beta
                                        * pixel_mut.vp.bsdf.as_ref().unwrap().f(
                                            &pixel_mut.vp.wo,
                                            &mut wi,
                                            BxDFType::all(),
                                        );
                                    for i in 0..Spectrum::n_samples() {
                                        pixel_mut.phi[i] += phi[i];
                                    }
                                    pixel_mut.m += 1;
                                    if let Some(next) = node.next {
                                        node_index = next;
                                    } else {
                                        break;
                                    }
                                }
                            }
                        }

                        isect.compute_scattering_functions(
                            &photon_ray,
                            true,
                            TransportMode::Importance,
                        );
                        if isect.bsdf.is_none() {
                            depth -= 1;
                            photon_ray = isect.spawn_ray(&photon_ray.d).into();
                            continue;
                        }

                        let photon_bsdf = isect.bsdf.as_ref().unwrap();

                        let mut wi = Vector3f::default();
                        let wo = -photon_ray.d;
                        let mut pdf = 0.0;
                        let mut flags = BxDFType::empty();

                        let bsdf_sample = Point2f::new(
                            radical_inverse(halton_dim, halton_index),
                            radical_inverse(halton_dim + 1, halton_index),
                        );
                        halton_dim += 2;
                        let fr = photon_bsdf.sample_f(
                            &wo,
                            &mut wi,
                            &bsdf_sample,
                            &mut pdf,
                            BxDFType::all(),
                            Some(&mut flags),
                        );
                        if fr.is_black() || pdf == 0.0 {
                            break;
                        }

                        let b_new = beta * fr * wi.abs_dot(&isect.shading.n) / pdf;

                        let q = (1.0 - b_new.y_value() / beta.y_value()).min(0.0);
                        if radical_inverse(halton_dim, halton_index) < q {
                            break;
                        }
                        halton_dim += 1;
                        beta = b_new / (1.0 - q);
                        photon_ray = isect.spawn_ray(&wi).into();

                        depth += 1;
                    }
                },
                self.photon_per_iteration,
                8192
            );

            parallel_for!(
                |i: usize| {
                    let mut pixel = pixels[i].clone();
                    let p = Arc::get_mut(&mut pixel).unwrap();
                    if p.m > 0 {
                        let gamma = 2.0 / 3.0;
                        let n_new = p.n + gamma * p.m as Float;
                        let r_new = p.radius * (n_new / (p.n + p.m as Float)).sqrt();
                        let mut phi = Spectrum::default();
                        for j in 0..Spectrum::n_samples() {
                            phi[j] = p.phi[j];
                        }
                        p.tau = (p.tau + phi * p.vp.beta) * (r_new * r_new) / (p.radius * p.radius);
                        p.n = n_new;
                        p.radius = r_new;
                        p.m = 0;
                        for j in 0..Spectrum::n_samples() {
                            p.phi[j] = 0.0;
                        }
                    }
                    p.vp.beta = Spectrum::new(0.0);
                    p.vp.bsdf = None;
                },
                n_pixels as usize,
                4096
            );

            if iter + 1 == self.n_iterations || iter + 1 % self.write_frequency == 0 {
                let x0 = pixel_bounds.min.x;
                let x1 = pixel_bounds.max.x;
                let np = (iter + 1) * self.photon_per_iteration;
                let mut image = vec![Spectrum::new(0.0); pixel_bounds.area() as usize];
                let mut offset = 0;
                for y in pixel_bounds.min.y..pixel_bounds.max.y {
                    for x in x0..x1 {
                        let pixel = pixels
                            [((y - pixel_bounds.min.y) * (x1 - x0) + (x - x0)) as usize]
                            .clone();
                        let mut l = pixel.ld / (iter + 1) as Float;
                        l += pixel.tau / (np as Float * PI * pixel.radius * pixel.radius);
                        image[offset] = l;
                        offset += 1;
                    }
                }

                self.camera
                    .film()
                    .write()
                    .unwrap()
                    .set_image(image.as_slice());
                self.camera.film().write().unwrap().write_image(1.0);

                //TODO
            }
        }
    }
}
