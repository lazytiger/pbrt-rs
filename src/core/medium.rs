use crate::core::{
    geometry::{Point2f, Ray, Vector3f},
    pbrt::{any_equal, Float},
    sampler::SamplerDtRw,
    spectrum::Spectrum,
};
use std::{
    any::Any,
    sync::{Arc, Mutex, RwLock},
};

pub trait PhaseFunction {
    fn as_any(&self) -> &dyn Any;
    fn p(&self, wo: &Vector3f, wi: &Vector3f) -> Float;
    fn sample_p(&self, wo: &Vector3f, wi: &mut Vector3f, u: &Point2f) -> Float;
    fn to_string(&self) -> String;
}

pub trait Medium {
    fn as_any(&self) -> &dyn Any;
    fn tr(&self, ray: &Ray, sampler: SamplerDtRw) -> Spectrum;
    fn sample(&self, ray: &Ray, sampler: SamplerDtRw, mi: &mut MediumInterface) -> Spectrum;
}

pub type MediumDt = Arc<Box<dyn Medium>>;
pub type MediumDtMut = Arc<Mutex<Box<dyn Medium>>>;
pub type MediumDtRw = Arc<RwLock<Box<dyn Medium>>>;

#[derive(Default, Clone)]
pub struct MediumInterface {
    pub inside: Option<MediumDt>,
    pub outside: Option<MediumDt>,
}

impl MediumInterface {
    pub fn new(inside: Option<MediumDt>, outside: Option<MediumDt>) -> MediumInterface {
        MediumInterface { inside, outside }
    }

    pub fn is_medium_transition(&self) -> bool {
        if self.inside.is_none() && self.outside.is_none() {
            false
        } else if self.inside.is_some() && self.outside.is_some() {
            let inside = self.inside.as_ref().unwrap();
            let outside = self.outside.as_ref().unwrap();
            !any_equal(inside.as_any(), outside.as_any())
        } else {
            true
        }
    }
}

pub struct HenyeyGreenstein {}

impl HenyeyGreenstein {
    pub fn p(&self, wo: &Vector3f, wi: &Vector3f) -> Float {
        todo!()
    }
    pub fn sample_p(&self, wo: &Vector3f, wi: &mut Vector3f, sample: &Point2f) -> Float {
        todo!()
    }
}
