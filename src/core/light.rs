use crate::core::{
    geometry::{Normal3f, Point2f, Ray, RayDifferentials, Vector3f},
    interaction::{Interaction, SurfaceInteraction},
    medium::MediumInterface,
    pbrt::Float,
    sampler::Sampler,
    scene::Scene,
    spectrum::Spectrum,
    transform::{Transform, Transformf},
};
use std::{
    any::Any,
    ops::{BitAnd, BitOr},
    sync::Arc,
};

#[derive(Copy, Clone)]
pub struct LightFlags(u8);

impl LightFlags {
    pub fn delta_position() -> LightFlags {
        LightFlags(1)
    }
    pub fn delta_direction() -> LightFlags {
        LightFlags(2)
    }
    pub fn area() -> LightFlags {
        LightFlags(4)
    }
    pub fn infinite() -> LightFlags {
        LightFlags(8)
    }
}

impl BitAnd for LightFlags {
    type Output = LightFlags;

    fn bitand(self, rhs: Self) -> Self::Output {
        LightFlags(self.0 & rhs.0)
    }
}

impl BitOr for LightFlags {
    type Output = LightFlags;

    fn bitor(self, rhs: Self) -> Self::Output {
        LightFlags(self.0 | rhs.0)
    }
}

impl Into<bool> for LightFlags {
    fn into(self) -> bool {
        self.0 != 0
    }
}

#[inline]
pub fn is_delta_light(flags: LightFlags) -> bool {
    (flags & LightFlags::delta_direction()).into() || (flags & LightFlags::delta_position()).into()
}

pub trait Light {
    fn as_any(&self) -> &dyn Any;
    fn sample_li(
        self,
        iref: &Interaction,
        u: &Point2f,
        wi: &mut Vector3f,
        pdf: &mut Float,
        vis: &mut VisibilityTester,
    ) -> Spectrum;
    fn power(&self) -> Spectrum;
    fn pre_process(&self, scene: &Scene) {}

    fn le(&self, r: &RayDifferentials) -> Spectrum {
        todo!()
    }

    fn pdf_li(&self, iref: &Interaction, wi: &Vector3f) -> Float;

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

pub struct VisibilityTester {
    p0: Interaction,
    p1: Interaction,
}

impl VisibilityTester {
    pub fn new(p0: Interaction, p1: Interaction) -> VisibilityTester {
        VisibilityTester { p0, p1 }
    }
    pub fn un_occluded(&self, scene: &Scene) -> bool {
        todo!()
    }
    pub fn tr(&self, scene: &Scene, sampler: Arc<Box<dyn Sampler>>) -> Spectrum {
        todo!()
    }
    pub fn p0(&self) -> &Interaction {
        &self.p0
    }

    pub fn p1(&self) -> &Interaction {
        &self.p1
    }
}
