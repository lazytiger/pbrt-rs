use crate::core::{
    geometry::{Normal3f, Vector3f},
    pbrt::{clamp, Float},
    spectrum::Spectrum,
};
use bitflags::bitflags;

pub fn fr_dielectric(cos_theta_i: Float, eta_i: Float, eta_t: Float) -> Float {
    todo!()
}

pub fn fr_conductor(
    cos_theta_i: Float,
    eta_i: &Spectrum,
    eta_t: &Spectrum,
    k: &Spectrum,
) -> Spectrum {
    todo!()
}

#[inline]
pub fn cos_theta(w: &Vector3f) -> Float {
    w.z
}
#[inline]
pub fn cos2_theta(w: &Vector3f) -> Float {
    w.z * w.z
}
#[inline]
pub fn abs_cos_theta(w: &Vector3f) -> Float {
    w.z.abs()
}
#[inline]
pub fn sin2_theta(w: &Vector3f) -> Float {
    (1.0 - cos2_theta(w)).max(0.0)
}
#[inline]
pub fn sin_theta(w: &Vector3f) -> Float {
    sin2_theta(w).sqrt()
}
#[inline]
pub fn tan_theta(w: &Vector3f) -> Float {
    sin_theta(w) / cos_theta(w)
}
#[inline]
pub fn tan2_theta(w: &Vector3f) -> Float {
    sin2_theta(w) / cos2_theta(w)
}
#[inline]
pub fn cos_phi(w: &Vector3f) -> Float {
    let sin_theta = sin_theta(w);
    if sin_theta == 0.0 {
        1.0
    } else {
        clamp(w.x / sin_theta, -1.0, 1.0)
    }
}
#[inline]
pub fn sin_phi(w: &Vector3f) -> Float {
    let sin_theta = sin_theta(w);
    if sin_theta == 0.0 {
        0.0
    } else {
        clamp(w.y / sin_theta, -1.0, 1.0)
    }
}
#[inline]
pub fn cos2_phi(w: &Vector3f) -> Float {
    cos_phi(w) * cos_phi(w)
}
#[inline]
pub fn sin2_phi(w: &Vector3f) -> Float {
    sin_phi(w) * sin_phi(w)
}
#[inline]
pub fn cos_d_phi(wa: Vector3f, wb: Vector3f) -> Float {
    let waxy = wa.x * wa.x + wa.y * wa.y;
    let wbxy = wb.x * wb.x + wb.y * wb.y;
    if waxy == 0.0 || wbxy == 0.0 {
        1.0
    } else {
        clamp(
            (wa.x * wb.x + wa.y * wb.y) / (waxy * wbxy).sqrt(),
            -1.0,
            1.0,
        )
    }
}
#[inline]
pub fn reflect(wo: &Vector3f, n: &Vector3f) -> Vector3f {
    -*wo + *n * (2.0 * wo.dot(n))
}

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
#[inline]
pub fn same_hemisphere(w: &Vector3f, wp: &Vector3f) -> bool {
    w.z * wp.z > 0.0
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

pub struct FourierBSDFTable {
    eta: Float,
    m_max: usize,
    n_channels: usize,
    n_mu: usize,
    mu: Vec<Float>,
    m: Vec<usize>,
    a_offset: Vec<usize>,
    a: Vec<Float>,
    a0: Vec<Float>,
    cdf: Vec<Float>,
    recip: Vec<Float>,
}

impl FourierBSDFTable {
    pub fn read(filename: String, table: &mut FourierBSDFTable) -> bool {
        todo!()
    }

    pub fn get_ak(&self, offset_i: usize, offset_o: usize) -> (&[usize], &[Float]) {
        (
            &self.m[(offset_o * self.n_mu + offset_i)..],
            &self.a[self.a_offset[offset_o * self.n_mu + offset_i]..],
        )
    }

    pub fn get_weights_and_offset(
        &self,
        cos_theta: Float,
        offset: &mut [usize],
        weights: &mut [Float],
    ) -> bool {
        todo!()
    }
}

pub struct BSDF {}
