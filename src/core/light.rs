use crate::core::{
    geometry::{Normal3f, Point2f, Ray, RayDifferentials, Vector3f},
    interaction::{Interaction, InteractionDt, SurfaceInteraction},
    medium::MediumInterface,
    pbrt::Float,
    sampler::{Sampler, SamplerDt, SamplerDtRw},
    scene::Scene,
    spectrum::Spectrum,
    transform::{Transform, Transformf},
};
use std::{
    any::Any,
    sync::{Arc, Mutex, RwLock},
};

bitflags::bitflags! {
    pub struct LightFlags:u8 {
        const DELTA_POSITION = 1;
        const DELTA_DIRECTION = 2;
        const AREA = 4;
        const INFINITE = 8;
    }
}

#[inline]
pub fn is_delta_light(flags: LightFlags) -> bool {
    (flags & LightFlags::DELTA_DIRECTION).is_empty()
        || (flags & LightFlags::DELTA_POSITION).is_empty()
}

pub trait Light {
    fn as_any(&self) -> &dyn Any;
    fn sample_li(
        &self,
        iref: InteractionDt,
        u: &Point2f,
        wi: &mut Vector3f,
        pdf: &mut Float,
        vis: &mut VisibilityTester,
    ) -> Spectrum;
    fn power(&self) -> Spectrum;
    fn pre_process(&self, _scene: &Scene) {}

    fn le(&self, _r: &RayDifferentials) -> Spectrum {
        todo!()
    }

    fn pdf_li(&self, iref: InteractionDt, wi: &Vector3f) -> Float;

    fn sample_le(
        &self,
        u1: &Point2f,
        u2: &Point2f,
        time: Float,
        ray: &mut Ray,
        n_light: &mut Normal3f,
        pdf_pos: &mut Float,
        pdf_dir: &mut Float,
    );

    fn flags(&self) -> LightFlags;

    fn pdf_le(&self, ray: &Ray, n_light: &Vector3f, pdf_pos: &mut Float, pdf_dir: &mut Float);
}

pub struct BaseLight {
    pub flags: LightFlags,
    pub n_samples: usize,
    pub medium_interface: MediumInterface,
    pub light_to_world: Transformf,
    pub world_to_light: Transformf,
}

pub trait AreaLight: Light {
    fn l(&self, si: &SurfaceInteraction, v: &Vector3f) -> Spectrum;
}

#[derive(Default)]
pub struct VisibilityTester {
    p0: Option<InteractionDt>,
    p1: Option<InteractionDt>,
}

impl VisibilityTester {
    pub fn new(p0: InteractionDt, p1: InteractionDt) -> VisibilityTester {
        VisibilityTester {
            p0: Some(p0),
            p1: Some(p1),
        }
    }
    pub fn un_occluded(&self, _scene: &Scene) -> bool {
        todo!()
    }
    pub fn tr(&self, _scene: &Scene, _sampler: SamplerDtRw) -> Spectrum {
        todo!()
    }
    pub fn p0(&self) -> InteractionDt {
        self.p0.as_ref().unwrap().clone()
    }

    pub fn p1(&self) -> InteractionDt {
        self.p1.as_ref().unwrap().clone()
    }
}

pub type LightDt = Arc<Box<dyn Light>>;
pub type LightDtMut = Arc<Mutex<Box<dyn Light>>>;
pub type LightDtRw = Arc<RwLock<Box<dyn Light>>>;
pub type AreaLightDt = Arc<Box<dyn AreaLight>>;
pub type AreaLightDtMut = Arc<Mutex<Box<dyn AreaLight>>>;
pub type AreaLightDtRw = Arc<RwLock<Box<dyn AreaLight>>>;
