use crate::core::{
    geometry::{Normal3f, Vector3f},
    pbrt::Float,
};
use bitflags::bitflags;


#[inline]
pub fn refract(wi: &Vector3f, n: &Normal3f, eta: Float, wt: &mut Vector3f) -> bool {
    let cos_theta_i = n.dot(wi);
    let sin2_theta_i = (1.0 - cos_theta_i * cos_theta_i).max(0.0);
    let sin2_theta_t = eta * eta * sin2_theta_i;

    if sin2_theta_i >= 1.0 {
        return false;
    }
    let cos_theta_t = (1.0 - sin2_theta_t).sqrt();
    *wt = -*wi * eta + *n * (eta * cos_theta_i - cos_theta_t);
    true
}

bitflags! {
    pub struct BxDFType:u8 {
        const  BSDF_REFLECTION = 1 << 0;
        const BSDF_TRANSMISSION = 1 << 1;
        const BSDF_DIFFUSE = 1 << 2;
        const BSDF_GLOSSY = 1 << 3;
        const BSDF_SPECULAR = 1 << 4;
    }
}
