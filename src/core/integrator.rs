use crate::core::{
    arena::Arena,
    camera::{Camera, CameraDt},
    geometry::{Bounds2i, Point2f, RayDifferentials, Vector3f},
    interaction::{Interaction, InteractionDt, MediumInteraction, SurfaceInteraction},
    light::{is_delta_light, Light, LightDt, VisibilityTester},
    pbrt::{any_equal, Float},
    reflection::{BxDF, BxDFType},
    sampler::{Sampler, SamplerDt, SamplerDtMut, SamplerDtRw},
    sampling::{power_heuristic, Distribution1D},
    scene::Scene,
    spectrum::{spectrum_lerp, Spectrum},
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
    it: InteractionDt,
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
                it.clone(),
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
                    it.clone(),
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
    it: InteractionDt,
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
    it: InteractionDt,
    u_scattering: &Point2f,
    light: LightDt,
    u_light: &Point2f,
    scene: &Scene,
    sampler: SamplerDtRw,
    handle_media: bool,
    specular: bool,
) -> Spectrum {
    let bsdf_flags = if specular {
        BxDFType::all()
    } else {
        BxDFType::all() ^ !BxDFType::BSDF_SPECULAR
    };
    let mut ld = Spectrum::new(0.0);
    let mut wi = Vector3f::default();
    let (mut light_pdf, mut scattering_pdf) = (0.0, 0.0);
    let mut visibility = VisibilityTester::default();
    let mut li = light.sample_li(
        it.clone(),
        u_light,
        &mut wi,
        &mut light_pdf,
        &mut visibility,
    );
    if light_pdf > 0.0 && !li.is_black() {
        let f = if it.is_surface_interaction() {
            let isect: &SurfaceInteraction = it.as_any().downcast_ref().unwrap();
            scattering_pdf = isect.bsdf.as_ref().unwrap().pdf(&isect.wo, &wi, bsdf_flags);
            isect.bsdf.as_ref().unwrap().f(&isect.wo, &wi, bsdf_flags)
                * wi.abs_dot(&isect.shading.n)
        } else {
            let mi: &MediumInteraction = it.as_any().downcast_ref().unwrap();
            let p = mi.phase.p(&mi.wo, &wi);
            scattering_pdf = p;
            Spectrum::new(p)
        };
        if !f.is_black() {
            if handle_media {
                li *= visibility.tr(scene, sampler.clone());
            } else {
                if !visibility.un_occluded(scene) {
                    li = Spectrum::new(0.0);
                } else {
                    log::debug!("shadow ray unoccluded");
                }
            }

            if !li.is_black() {
                if is_delta_light(light.flags()) {
                    ld += li * f / light_pdf;
                } else {
                    let weight = power_heuristic(1, light_pdf, 1, scattering_pdf);
                    ld += li * f * weight / light_pdf;
                }
            }
        }
    }

    if !is_delta_light(light.flags()) {
        let mut sampled_specular = false;
        let f = if it.is_surface_interaction() {
            let isect: &SurfaceInteraction = it.as_any().downcast_ref().unwrap();
            let mut sampled_type = BxDFType::empty();
            let mut f = isect.bsdf.as_ref().unwrap().sample_f(
                &isect.wo,
                &mut wi,
                u_scattering,
                &mut scattering_pdf,
                bsdf_flags,
                &mut sampled_type,
            );
            f *= wi.abs_dot(&isect.shading.n);
            sampled_specular = !(sampled_type & BxDFType::BSDF_SPECULAR).is_empty();
            f
        } else {
            let mi: &MediumInteraction = it.as_any().downcast_ref().unwrap();
            let p = mi.phase.sample_p(&mi.wo, &mut wi, u_scattering);
            scattering_pdf = p;
            Spectrum::new(p)
        };

        if !f.is_black() && scattering_pdf > 0.0 {
            let mut weight = 1.0;
            if !sampled_specular {
                light_pdf = light.pdf_li(it.clone(), &wi);
                if light_pdf == 0.0 {
                    return ld;
                }
                weight = power_heuristic(1, scattering_pdf, 1, light_pdf);
            }

            let mut light_isect = SurfaceInteraction::default();
            let ray = it.as_base().spawn_ray(&wi);
            let mut tr = Spectrum::new(1.0);
            let found_surface_interaction = if handle_media {
                scene.intersect_tr(&ray, sampler.clone(), &mut light_isect, &mut tr)
            } else {
                scene.intersect(&ray, &mut light_isect)
            };

            let mut li = Spectrum::new(0.0);
            if found_surface_interaction {
                if any_equal(
                    light_isect
                        .primitive
                        .as_ref()
                        .unwrap()
                        .get_area_light()
                        .unwrap()
                        .as_any(),
                    light.as_any(),
                ) {
                    li = light_isect.le(&-wi);
                } else {
                    li = light.le(&ray.into());
                }
            }
            if !li.is_black() {
                ld += li * f * tr * weight / scattering_pdf;
            }
        }
    }

    ld
}

pub fn compute_light_power_distribution(scene: &Scene) -> Option<Box<Distribution1D>> {
    if scene.lights.is_empty() {
        return None;
    }
    let mut light_power = Vec::new();
    for light in &scene.lights {
        light_power.push(light.power().y_value());
    }
    Some(Box::new(Distribution1D::new(light_power.as_slice())))
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

impl Integrator for BaseSamplerIntegrator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn render(&self, scene: &Scene) {
        todo!()
    }
}
