use crate::core::{
    arena::Arena,
    camera::{Camera, CameraDt},
    geometry::{Bounds2i, Point2f, RayDifferentials},
    interaction::{Interaction, SurfaceInteraction},
    light::{Light, LightDt},
    pbrt::Float,
    reflection::BxDFType,
    sampler::{Sampler, SamplerDt, SamplerDtMut, SamplerDtRw},
    sampling::Distribution1D,
    scene::Scene,
    spectrum::Spectrum,
};
use std::{
    any::Any,
    io::Write,
    sync::{Arc, Mutex, RwLock},
};

pub type IntegratorDt = Arc<Box<dyn Integrator>>;
pub type IntegratorDtMut = Arc<Mutex<Box<dyn Integrator>>>;
pub type IntegratorDtRw = Arc<RwLock<Box<dyn Integrator>>>;
pub type SamplerIntegratorDt = Arc<Box<dyn SamplerIntegrator>>;
pub type SamplerIntegratorDtMut = Arc<Mutex<Box<dyn SamplerIntegrator>>>;
pub type SamplerIntegratorDtRw = Arc<RwLock<Box<dyn SamplerIntegrator>>>;

pub trait Integrator {
    fn as_any(&self) -> &dyn Any;
    fn render(&self, scene: &Scene);
}

pub fn uniform_sample_all_lights(
    it: &Interaction,
    scene: &Scene,
    sampler: SamplerDtRw,
    n_light_samples: Vec<usize>,
    handle_media: bool,
) {
    let mut l = Spectrum::new(0.0);
    for j in 0..scene.lights.len() {
        let light = scene.lights[j].clone();
        let n_samples = n_light_samples[j];
        let u_light_array = sampler.write().unwrap().get_2d_array(n_samples);
        let u_scattering_array = sampler.write().unwrap().get_2d_array(n_samples);
        if u_light_array.is_none() || u_scattering_array.is_none() {
            let u_light = sampler.write().unwrap().get_2d();
            let u_scattering = sampler.write().unwrap().get_2d();
            l += estimate_direct(
                it,
                &u_scattering,
                light,
                &u_light,
                scene,
                sampler.clone(),
                handle_media,
                false,
            );
        } else {
            let u_light_array = u_light_array.unwrap();
            let u_scattering_array = u_scattering_array.unwrap();
            let mut ld = Spectrum::new(0.0);
            for k in 0..n_samples {
                ld += estimate_direct(
                    it,
                    &u_scattering_array[k],
                    light.clone(),
                    &u_light_array[k],
                    scene,
                    sampler.clone(),
                    handle_media,
                    false,
                );
            }
        }
    }
    todo!()
}

pub fn uniform_sample_one_light(
    it: &Interaction,
    scene: &Scene,
    sampler: SamplerDtRw,
    handle_media: bool,
    light_distrib: Option<&Distribution1D>,
) -> Spectrum {
    let n_lights = scene.lights.len();
    if n_lights == 0 {
        return Spectrum::new(0.0);
    }

    let mut light_num = 0;
    let mut light_pdf = 0.0;
    if let Some(light_distrib) = light_distrib {
        light_num = light_distrib.sample_discrete(
            sampler.write().unwrap().get_1d(),
            Some(&mut light_pdf),
            None,
        );
        if light_pdf == 0.0 {
            return Spectrum::new(0.0);
        }
    } else {
        light_num = (sampler.write().unwrap().get_1d() * n_lights as Float)
            .min(n_lights as Float - 1.0) as usize;
        light_pdf = 1.0 / n_lights as Float;
    }

    let light = scene.lights[light_num].clone();
    let u_light = sampler.write().unwrap().get_2d();
    let u_scattering = sampler.write().unwrap().get_2d();
    estimate_direct(
        it,
        &u_scattering,
        light,
        &u_light,
        scene,
        sampler,
        handle_media,
        false,
    ) / light_pdf
}

pub fn estimate_direct(
    _it: &Interaction,
    _u_shading: &Point2f,
    _light: LightDt,
    _u_light: &Point2f,
    _scene: &Scene,
    _sampler: SamplerDtRw,
    _handle_media: bool,
    specular: bool,
) -> Spectrum {
    let _bsdf_flags = if specular {
        BxDFType::all()
    } else {
        BxDFType::all() ^ !BxDFType::BSDF_SPECULAR
    };
    let _ld = Spectrum::new(0.0);
    todo!()
}

pub fn compute_light_power_distribution(_scene: &Scene) -> Box<Distribution1D> {
    todo!()
}

pub trait SamplerIntegrator: Integrator {
    fn pre_process(&self, _scene: &Scene, _sampler: SamplerDt) {}
    fn li(&self, ray: &RayDifferentials, scene: &Scene, sampler: SamplerDt, depth: i32);
}

pub struct BaseSamplerIntegrator {
    pub camera: CameraDt,
    sampler: SamplerDt,
    pixel_bounds: Bounds2i,
}

impl BaseSamplerIntegrator {
    pub fn new(camera: CameraDt, sampler: SamplerDt, pixel_bounds: Bounds2i) -> Self {
        Self {
            camera,
            sampler,
            pixel_bounds,
        }
    }

    pub fn specular_reflect(
        &self,
        _ray: &RayDifferentials,
        _isect: &SurfaceInteraction,
        _scene: &Scene,
        _sampler: SamplerDt,
        _depth: i32,
    ) {
        todo!()
    }

    pub fn specular_transmit(
        &self,
        _ray: &RayDifferentials,
        _isect: &SurfaceInteraction,
        _scene: &Scene,
        _sampler: SamplerDt,
        _depth: i32,
    ) {
        todo!()
    }
}
