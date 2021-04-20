use crate::core::{
    geometry::{Point2f, Vector3f},
    interaction::SurfaceInteraction,
    pbrt::Float,
    scene::Scene,
    spectrum::Spectrum,
};
use std::sync::{Arc, Mutex, RwLock};

pub fn fresnel_moment1(inv_eta: Float) -> Float {
    todo!()
}

pub fn fresnel_moment2(inv_eta_: Float) -> Float {
    todo!()
}

pub type BSSRDFDt = Arc<Box<dyn BSSRDF + Sync + Send>>;
pub type BSSRDFDtMut = Arc<Mutex<Box<dyn BSSRDF + Sync + Send>>>;
pub type BSSRDFDtRw = Arc<RwLock<Box<dyn BSSRDF + Sync + Send>>>;

pub trait BSSRDF {
    fn s(&self, pi: &SurfaceInteraction, wi: &Vector3f) -> Spectrum;
    fn sample_s(
        &self,
        scene: &Scene,
        u1: Float,
        u2: &Point2f,
        si: &mut SurfaceInteraction,
        pdf: &mut Float,
    ) -> Spectrum;
}
