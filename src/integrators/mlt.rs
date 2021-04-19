use crate::{
    core::{
        camera::CameraDt,
        geometry::{Bounds2f, Bounds2i, Point2f, Point2i, RayDifferentials},
        integrator::{compute_light_power_distribution, Integrator, SamplerIntegrator},
        pbrt::{clamp, erf_inv, Float},
        primitive::Primitive,
        rng::RNG,
        sampler::{BaseSampler, Sampler, SamplerDtRw},
        sampling::Distribution1D,
        scene::Scene,
        spectrum::Spectrum,
    },
    impl_base_sampler,
    integrators::bdpt::{
        connect_bdpt, generate_camera_sub_path, generate_light_sub_path, LightKey, Vertex,
    },
    parallel_for,
};
use derive_more::{Deref, DerefMut};
use num::integer::Roots;
use std::{
    any::Any,
    collections::HashMap,
    f32::consts::SQRT_2,
    hash::Hash,
    sync::{Arc, RwLock},
};

#[derive(Default, Clone)]
pub struct PrimarySample {
    value: Float,
    last_modification_iteration: usize,
    value_backup: Float,
    modify_backup: usize,
}

impl PrimarySample {
    fn backup(&mut self) {
        self.value_backup = self.value;
        self.modify_backup = self.last_modification_iteration;
    }
    fn restore(&mut self) {
        self.value = self.value_backup;
        self.last_modification_iteration = self.modify_backup;
    }
}

#[derive(Deref, DerefMut)]
pub struct MLTSampler {
    #[deref]
    #[deref_mut]
    base: BaseSampler,
    rng: RNG,
    sigma: Float,
    large_step_probability: Float,
    stream_count: usize,
    x: Vec<PrimarySample>,
    current_iteration: usize,
    large_step: bool,
    last_large_step_iteration: usize,
    stream_index: usize,
    sample_index: usize,
}

impl MLTSampler {
    pub fn new(
        mutations_per_pixel: i64,
        rng_sequence_index: usize,
        sigma: Float,
        large_step_probability: Float,
        stream_count: usize,
    ) -> Self {
        Self {
            base: BaseSampler::new(mutations_per_pixel),
            rng: RNG::new(rng_sequence_index),
            sigma,
            large_step_probability,
            stream_count,
            x: vec![],
            current_iteration: 0,
            large_step: true,
            last_large_step_iteration: 0,
            stream_index: 0,
            sample_index: 0,
        }
    }

    fn start_iteration(&mut self) {
        self.current_iteration += 1;
        self.large_step = self.rng.uniform_float() < self.large_step_probability
    }

    fn accept(&mut self) {
        if self.large_step {
            self.last_large_step_iteration = self.current_iteration;
        }
    }

    fn reject(&mut self) {
        for xi in &mut self.x {
            if xi.last_modification_iteration == self.current_iteration {
                xi.restore();
            }
            self.current_iteration -= 1;
        }
    }

    fn start_stream(&mut self, index: usize) {
        self.stream_index = index;
        self.sample_index = 0;
    }

    fn get_next_index(&mut self) -> usize {
        self.sample_index += 1;
        self.stream_index + self.stream_count * (self.sample_index - 1)
    }

    fn ensure_ready(&mut self, index: usize) {
        if index >= self.x.len() {
            self.x.resize(index + 1, PrimarySample::default());
        }
        let mut xi = &mut self.x[index];
        if xi.last_modification_iteration < self.last_large_step_iteration {
            xi.value = self.rng.uniform_float();
            xi.last_modification_iteration = self.last_large_step_iteration;
        }

        xi.backup();
        if self.large_step {
            xi.value = self.rng.uniform_float();
        } else {
            let n_small = self.current_iteration - xi.last_modification_iteration;
            let normal_sample = SQRT_2 * erf_inv(2.0 * self.rng.uniform_float() - 1.0);
            let eff_sigma = self.sigma * (n_small as Float).sqrt();
            xi.value += normal_sample * eff_sigma;
            xi.value -= xi.value.floor();
        }
        xi.last_modification_iteration = self.current_iteration;
    }
}

impl Sampler for MLTSampler {
    impl_base_sampler!();

    fn get_1d(&mut self) -> f32 {
        let index = self.get_next_index();
        self.ensure_ready(index);
        self.x[index].value
    }

    fn get_2d(&mut self) -> Point2f {
        Point2f::new(self.get_1d(), self.get_1d())
    }

    fn clone_sampler(&self, seed: usize) -> SamplerDtRw {
        unimplemented!()
    }
}

pub struct MLTIntegrator {
    camera: CameraDt,
    max_depth: usize,
    n_bootstrap: usize,
    n_chains: usize,
    mutation_per_pixel: usize,
    sigma: Float,
    large_step_probability: Float,
}

impl MLTIntegrator {
    pub fn new(
        camera: CameraDt,
        max_depth: usize,
        n_bootstrap: usize,
        n_chains: usize,
        mutation_per_pixel: usize,
        sigma: Float,
        large_step_probability: Float,
    ) -> Self {
        Self {
            camera,
            max_depth,
            n_bootstrap,
            n_chains,
            mutation_per_pixel,
            sigma,
            large_step_probability,
        }
    }

    fn l(
        &self,
        scene: &Scene,
        light_distr: &Distribution1D,
        light2index: &HashMap<LightKey, usize>,
        sampler: SamplerDtRw,
        depth: usize,
        p_raster: &mut Point2f,
    ) -> Spectrum {
        sampler
            .write()
            .unwrap()
            .as_mut_any()
            .downcast_mut::<MLTSampler>()
            .unwrap()
            .start_stream(CAMERA_STREAM_INDEX);
        let (mut s, mut t, mut n_strategies) = (0, 0, 0);
        if depth == 0 {
            n_strategies = 1;
            s = 0;
            t = 2;
        } else {
            n_strategies = depth + 2;
            s = std::cmp::min(
                sampler.write().unwrap().get_1d() as usize * n_strategies,
                n_strategies - 1,
            );
            t = n_strategies - s;
        }
        let mut camera_vertices = vec![Vertex::default(); t];
        let sample_bounds: Bounds2f = self
            .camera
            .film()
            .read()
            .unwrap()
            .get_sample_bounds()
            .into();

        *p_raster = sample_bounds.lerp(&sampler.write().unwrap().get_2d());
        if generate_camera_sub_path(
            scene,
            sampler.clone(),
            t,
            self.camera.clone(),
            *p_raster,
            camera_vertices.as_mut_slice(),
            0,
        ) != t
        {
            return 0.0.into();
        }

        sampler
            .write()
            .unwrap()
            .as_mut_any()
            .downcast_mut::<MLTSampler>()
            .unwrap()
            .start_stream(LIGHT_STREAM_INDEX);
        let mut light_vertices = vec![Vertex::default(); s];
        if generate_light_sub_path(
            scene,
            sampler.clone(),
            s,
            camera_vertices[0].time(),
            light_distr,
            light2index,
            light_vertices.as_mut_slice(),
            0,
        ) != s
        {
            return 0.0.into();
        }

        sampler
            .write()
            .unwrap()
            .as_mut_any()
            .downcast_mut::<MLTSampler>()
            .unwrap()
            .start_stream(CONNECTION_STREAM_INDEX);
        connect_bdpt(
            scene,
            light_vertices.as_mut_slice(),
            camera_vertices.as_mut_slice(),
            s,
            t,
            light_distr,
            light2index,
            self.camera.clone(),
            sampler.clone(),
            p_raster,
            None,
        ) * n_strategies as Float
    }
}

impl Integrator for MLTIntegrator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn render(&mut self, scene: &Scene) {
        let light_distr = compute_light_power_distribution(scene);
        let mut light2index = HashMap::new();
        for i in 0..scene.lights.len() {
            light2index.insert(LightKey(scene.lights[i].clone()), i);
        }

        let n_bootstrap_samples = self.n_bootstrap * (self.max_depth + 1);
        let mut bootstrap_weights = RwLock::new(vec![0.0; n_bootstrap_samples]);
        if scene.lights.len() > 0 {
            let chunk_size = clamp(self.n_bootstrap as isize / 128, 1, 8192) as usize;
            parallel_for!(
                |i: usize| {
                    for depth in 0..self.max_depth + 1 {
                        let rng_index = i * (self.max_depth + 1) + depth;
                        let sampler: SamplerDtRw =
                            Arc::new(RwLock::new(Box::new(MLTSampler::new(
                                self.mutation_per_pixel as i64,
                                rng_index,
                                self.sigma,
                                self.large_step_probability,
                                N_SAMPLE_STREAMS,
                            ))));
                        let mut p_raster = Point2f::default();
                        bootstrap_weights.write().unwrap()[rng_index] = self
                            .l(
                                scene,
                                light_distr.as_ref().unwrap(),
                                &light2index,
                                sampler,
                                depth,
                                &mut p_raster,
                            )
                            .y_value();
                    }
                },
                self.n_bootstrap,
                chunk_size,
            );
        }

        let bootstrap = Distribution1D::new(
            &bootstrap_weights.read().unwrap().as_slice()[..n_bootstrap_samples],
        );
        let b = bootstrap.func_int * (self.max_depth + 1) as Float;

        let film = self.camera.film().clone();
        let n_total_mutations =
            self.mutation_per_pixel * film.read().unwrap().get_sample_bounds().area() as usize;
        if scene.lights.len() > 0 {
            parallel_for!(
                |i| {
                    let n_chain_mutations = std::cmp::min(
                        (i + 1) * n_total_mutations / self.n_chains,
                        n_total_mutations,
                    ) - i * n_total_mutations / self.n_chains;
                    let mut rng = RNG::new(i);
                    let bootstrap_index =
                        bootstrap.sample_discrete(rng.uniform_float(), None, None);
                    let depth = bootstrap_index % (self.max_depth + 1);

                    let sampler: SamplerDtRw = Arc::new(RwLock::new(Box::new(MLTSampler::new(
                        self.mutation_per_pixel as i64,
                        bootstrap_index,
                        self.sigma,
                        self.large_step_probability,
                        N_SAMPLE_STREAMS,
                    ))));
                    let mut p_current = Point2f::default();
                    let mut l_current = self.l(
                        scene,
                        light_distr.as_ref().unwrap(),
                        &light2index,
                        sampler.clone(),
                        depth,
                        &mut p_current,
                    );

                    for j in 0..n_chain_mutations {
                        sampler
                            .write()
                            .unwrap()
                            .as_mut_any()
                            .downcast_mut::<MLTSampler>()
                            .unwrap()
                            .start_iteration();
                        let mut p_proposed = Point2f::default();
                        let l_proposed = self.l(
                            scene,
                            light_distr.as_ref().unwrap(),
                            &light2index,
                            sampler.clone(),
                            depth,
                            &mut p_proposed,
                        );
                        let accept = (l_proposed.y_value() / l_current.y_value()).min(1.0);

                        if accept > 0.0 {
                            film.write()
                                .unwrap()
                                .add_splat(&p_proposed, l_proposed * accept / l_proposed.y_value());
                        }
                        film.write().unwrap().add_splat(
                            &p_current,
                            l_current * (1.0 - accept) / l_current.y_value(),
                        );

                        if rng.uniform_float() < accept {
                            p_current = p_proposed;
                            l_current = l_proposed;
                            sampler
                                .write()
                                .unwrap()
                                .as_mut_any()
                                .downcast_mut::<MLTSampler>()
                                .unwrap()
                                .accept();
                        } else {
                            sampler
                                .write()
                                .unwrap()
                                .as_mut_any()
                                .downcast_mut::<MLTSampler>()
                                .unwrap()
                                .reject();
                        }
                    }
                },
                self.n_chains,
            );
        }
        self.camera
            .film()
            .write()
            .unwrap()
            .write_image(b / self.mutation_per_pixel as Float);
    }
}

const CAMERA_STREAM_INDEX: usize = 0;
const LIGHT_STREAM_INDEX: usize = 1;
const CONNECTION_STREAM_INDEX: usize = 2;
const N_SAMPLE_STREAMS: usize = 3;
