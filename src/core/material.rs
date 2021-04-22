use crate::core::{
    geometry::Vector2f, interaction::SurfaceInteraction, pbrt::Float, texture::TextureDt,
};
use std::{
    any::Any,
    fmt::Debug,
    sync::{Arc, Mutex, RwLock},
};

#[derive(Clone, Copy)]
pub enum TransportMode {
    Radiance,
    Importance,
}

pub trait Material: Debug {
    fn as_any(&self) -> &dyn Any;
    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceInteraction,
        mode: TransportMode,
        allow_multiple_lobes: bool,
    );
    fn bump(&self, d: TextureDt<Float>, si: &mut SurfaceInteraction) {
        let mut si_eval = si.clone();
        let mut du = 0.5 * (si.dudx.abs() + si.dudy.abs());
        if du == 0.0 {
            du = 0.0005;
        }
        si_eval.p = si.p + si.shading.dpdu * du;
        si_eval.uv = si.uv + Vector2f::new(du, 0.0);
        si_eval.n = (si.shading.dpdu.cross(&si.shading.dpdv) + si.dndu * du).normalize();
        let u_displace = d.evaluate(&si_eval);
        let mut dv = 0.5 * (si.dvdx.abs() + si.dvdy.abs());
        if dv == 0.0 {
            dv = 0.0005;
        }
        si_eval.p = si.p + si.shading.dpdv * dv;
        si_eval.uv = si.uv + Vector2f::new(0.0, dv);
        si_eval.n = (si.shading.dpdu.cross(&si.shading.dpdv) + si.dndv * dv).normalize();

        let v_displace = d.evaluate(&si_eval);
        let displace = d.evaluate(si);

        let dpdu = si.shading.dpdu
            + si.shading.n * ((v_displace - displace) / du)
            + si.shading.dndu * displace;

        let dpdv = si.shading.dpdv
            + si.shading.n * ((v_displace - displace) / dv)
            + si.shading.dndv * displace;

        si.set_shading_geometry(dpdu, dpdv, si.shading.dndu, si.shading.dndv, false);
    }
}

pub type MaterialDt = Arc<Box<dyn Material + Sync + Send>>;
pub type MaterialDtMut = Arc<Mutex<Box<dyn Material + Sync + Send>>>;
pub type MaterialDtRw = Arc<RwLock<Box<dyn Material + Sync + Send>>>;
