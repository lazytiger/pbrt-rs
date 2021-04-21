use crate::{
    core::{
        geometry::{Normal3f, Point2f, Vector3f},
        interaction::SurfaceInteraction,
        interpolation::{catmull_rom_weights, fourier, sample_catmull_rom_2d, sample_fourier},
        material::TransportMode,
        microfacet::MicrofacetDistributionDt,
        pbrt::{any_equal, clamp, radians, Float, INV_PI, ONE_MINUS_EPSILON, PI},
        sampling::{cosine_sample_hemisphere, uniform_hemisphere_pdf, uniform_sample_hemisphere},
        spectrum::{Spectrum, SpectrumType},
    },
    integrators::path::PathIntegrator,
};
use bitflags::bitflags;
use core::num::FpCategory::Normal;
use log::Level::Trace;
use std::{any::Any, sync::Arc};

pub fn fr_dielectric(cos_theta_i: Float, mut eta_i: Float, mut eta_t: Float) -> Float {
    let mut cos_theta_i = clamp(cos_theta_i, -1.0, 1.0);
    let entering = cos_theta_i > 0.0;
    if !entering {
        std::mem::swap(&mut eta_i, &mut eta_t);
        cos_theta_i = cos_theta_i.abs();
    }

    let sin_theta_i = (1.0 - cos_theta_i * cos_theta_i).max(0.0).sqrt();
    let sin_tehta_t = eta_i / eta_t * sin_theta_i;

    if sin_tehta_t >= 1.0 {
        return 1.0;
    }

    let cos_theta_t = (1.0 - sin_tehta_t * sin_tehta_t).max(0.0).sqrt();
    let r_parl =
        (eta_t * cos_theta_i - eta_i * cos_theta_t) / (eta_t * cos_theta_i + eta_i * cos_theta_t);
    let r_perp =
        (eta_i * cos_theta_i - eta_t * cos_theta_t) / (eta_i * cos_theta_i + eta_t * cos_theta_t);
    (r_parl * r_parl + r_perp * r_perp) / 2.0
}

pub fn fr_conductor(
    cos_theta_i: Float,
    eta_i: &Spectrum,
    eta_t: &Spectrum,
    k: &Spectrum,
) -> Spectrum {
    let cos_theta_i = clamp(cos_theta_i, -1.0, 1.0);
    let eta = eta_t / eta_i;
    let eta_k = k / eta_i;

    let cos_theta_i2 = cos_theta_i * cos_theta_i;
    let sin_theta_i2 = 1.0 - cos_theta_i2;
    let eta2 = eta * eta;
    let eta_k2 = eta_k * eta_k;

    let t0 = eta2 - eta_k2 - sin_theta_i2.into();
    let a2_plus_b2 = (t0 * t0 + eta2 * eta_k2 * 4.0).sqrt();
    let t1 = a2_plus_b2 + cos_theta_i2.into();
    let a = ((a2_plus_b2 + t0) * 0.5).sqrt();
    let t2 = a * (2.0 * cos_theta_i);
    let rs = (t1 - t2) / (t1 + t2);

    let t3 = a2_plus_b2 * cos_theta_i2 + (sin_theta_i2 * sin_theta_i2).into();
    let t4 = t2 * sin_theta_i2;
    let rp = rs * (t3 - t4) / (t3 + t4);

    (rp + rs) * 0.5
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

    pub fn get_ak(&self, offset_i: usize, offset_o: usize, m_ptr: &mut usize) -> &[Float] {
        *m_ptr = self.m[(offset_o * self.n_mu + offset_i)];
        &self.a[self.a_offset[offset_o * self.n_mu + offset_i]..]
    }

    pub fn get_weights_and_offset(
        &self,
        cos_theta: Float,
        offset: &mut usize,
        weights: &mut [Float],
    ) -> bool {
        catmull_rom_weights(self.n_mu, self.mu.as_slice(), cos_theta, offset, weights)
    }
}

#[derive(Clone)]
pub struct BSDF {
    pub eta: Float,
    ns: Normal3f,
    ng: Normal3f,
    ss: Vector3f,
    ts: Vector3f,
    n_bxdfs: usize,
    bxdfs: Vec<Option<BxDFDt>>,
}

impl BSDF {
    const MAX_BXDFS: usize = 8;

    pub fn new(si: &SurfaceInteraction, eta: Float) -> Self {
        let ns = si.shading.n;
        let ng = si.n;
        let ss = si.shading.dpdu.normalize();
        let ts = ns.cross(&ss);
        Self {
            eta,
            ns,
            ng,
            ss,
            ts,
            n_bxdfs: 0,
            bxdfs: vec![None; BSDF::MAX_BXDFS],
        }
    }

    pub fn add(&mut self, b: BxDFDt) {
        self.bxdfs[self.n_bxdfs] = Some(b);
        self.n_bxdfs += 1;
    }

    pub fn num_components(&self, flags: BxDFType) -> usize {
        let mut num = 0;
        for i in 0..self.n_bxdfs {
            if let Some(bxdf) = &self.bxdfs[i] {
                if bxdf.matches_flags(flags) {
                    num += 1;
                }
            }
        }
        num
    }

    pub fn world_to_local(&self, v: &Vector3f) -> Vector3f {
        Vector3f::new(v.dot(&self.ss), v.dot(&self.ts), v.dot(&self.ns))
    }

    pub fn local_to_world(&self, v: &Vector3f) -> Vector3f {
        Vector3f::new(
            self.ss.x * v.x + self.ts.x * v.y + self.ns.x * v.z,
            self.ss.y * v.x + self.ts.y * v.y * self.ns.y * v.z,
            self.ss.z * v.x + self.ts.z * v.y + self.ns.z * v.z,
        )
    }

    pub fn f(&self, wo_w: &Vector3f, wi_w: &Vector3f, flags: BxDFType) -> Spectrum {
        let wi = self.world_to_local(wi_w);
        let wo = self.world_to_local(wo_w);
        if wo.z == 0.0 {
            return 0.0.into();
        }
        let reflect = wi_w.dot(&self.ng) * wo_w.dot(&self.ng) > 0.0;
        let mut f = Spectrum::default();
        for i in 0..self.n_bxdfs {
            if let Some(bxdf) = &self.bxdfs[i] {
                if bxdf.matches_flags(flags)
                    && (reflect && bxdf.typ().contains(BxDFType::BSDF_REFLECTION)
                        || !reflect && bxdf.typ().contains(BxDFType::BSDF_TRANSMISSION))
                {
                    f += bxdf.f(&wo, &wi);
                }
            }
        }
        f
    }

    pub fn sample_f(
        &self,
        wo_w: &Vector3f,
        wi_w: &mut Vector3f,
        u: &Point2f,
        pdf: &mut f32,
        typ: BxDFType,
        mut sample_type: Option<&mut BxDFType>,
    ) -> Spectrum {
        let matching_comps = self.num_components(typ);
        if matching_comps == 0 {
            *pdf = 0.0;
            if let Some(sample_type) = sample_type {
                *sample_type = BxDFType::empty();
            }
            return 0.0.into();
        }

        let comp = std::cmp::min(
            (u[0] * matching_comps as Float).floor() as usize,
            matching_comps - 1,
        );

        let mut count = comp as isize;
        let mut bxdf = None;
        for i in 0..self.n_bxdfs {
            if let Some(_bxdf) = &self.bxdfs[i] {
                if _bxdf.matches_flags(typ) && {
                    let ret = count == 0;
                    count -= 1;
                    ret
                } {
                    bxdf.replace(_bxdf);
                    break;
                }
            }
        }
        let bxdf = bxdf.unwrap();

        let u_rempped = Point2f::new(
            ONE_MINUS_EPSILON.min(u[0] * matching_comps as Float - comp as Float),
            u[1],
        );
        let mut wi = Vector3f::default();
        let wo = self.world_to_local(wo_w);
        if wo.z == 0.0 {
            return 0.0.into();
        }
        *pdf = 0.0;
        if let Some(sample_type) = &mut sample_type {
            **sample_type = bxdf.typ();
        }

        let mut f = bxdf.sample_f(
            &wo,
            &mut wi,
            &u_rempped,
            pdf,
            Some(*sample_type.as_mut().unwrap()),
        );
        if *pdf == 0.0 {
            if let Some(sample_type) = sample_type {
                *sample_type = BxDFType::empty();
                return 0.0.into();
            }
        }
        *wi_w = self.local_to_world(&wi);
        if !bxdf.typ().contains(BxDFType::BSDF_SPECULAR) && matching_comps > 1 {
            for i in 0..self.n_bxdfs {
                if let Some(_bxdf) = &self.bxdfs[i] {
                    if !any_equal(bxdf.as_any(), _bxdf.as_any()) && _bxdf.matches_flags(typ) {
                        *pdf += _bxdf.pdf(&wo, &wi);
                    }
                }
            }
        }
        if matching_comps > 1 {
            *pdf /= matching_comps as Float;
        }

        if !bxdf.typ().contains(BxDFType::BSDF_SPECULAR) {
            let reflect = wi_w.dot(&self.ng) * wo_w.dot(&self.ng) > 0.0;
            f = Spectrum::new(0.0);
            for i in 0..self.n_bxdfs {
                if let Some(bxdf) = &self.bxdfs[i] {
                    if bxdf.matches_flags(typ)
                        && (reflect && bxdf.typ().contains(BxDFType::BSDF_REFLECTION)
                            || !reflect && bxdf.typ().contains(BxDFType::BSDF_TRANSMISSION))
                    {
                        f += bxdf.f(&wo, &wi);
                    }
                }
            }
        }
        f
    }

    pub fn rho(
        &self,
        wo_w: &Vector3f,
        n_samples: usize,
        samples: &[Point2f],
        flags: BxDFType,
    ) -> Spectrum {
        let wo = self.world_to_local(wo_w);
        let mut ret = Spectrum::new(0.0);
        for i in 0..self.n_bxdfs {
            if let Some(bxdf) = &self.bxdfs[i] {
                if bxdf.matches_flags(flags) {
                    ret += bxdf.rho(&wo, n_samples, samples);
                }
            }
        }
        ret
    }

    pub fn rho2(
        &self,
        n_samples: usize,
        samples1: &[Point2f],
        samples2: &[Point2f],
        flags: BxDFType,
    ) -> Spectrum {
        let mut ret = Spectrum::default();
        for i in 0..self.n_bxdfs {
            if let Some(bxdf) = &self.bxdfs[i] {
                if bxdf.matches_flags(flags) {
                    ret += bxdf.rho2(n_samples, samples1, samples2);
                }
            }
        }
        ret
    }

    pub fn pdf(&self, wo: &Vector3f, wi: &Vector3f, flags: BxDFType) -> f32 {
        if self.n_bxdfs == 0 {
            return 0.0;
        }

        let wo = self.world_to_local(wo);
        let wi = self.world_to_local(wi);
        if wo.z == 0.0 {
            return 0.0;
        }

        let mut pdf = 0.0;
        let mut matching_comps = 0;
        for i in 0..self.n_bxdfs {
            if let Some(bxdf) = &self.bxdfs[i] {
                if bxdf.matches_flags(flags) {
                    matching_comps += 1;
                    pdf += bxdf.pdf(&wo, &wi);
                }
            }
        }

        let v = if matching_comps > 0 {
            pdf / matching_comps as Float
        } else {
            0.0
        };
        v
    }
}

pub type BxDFDt = Arc<Box<dyn BxDF + Sync + Send>>;

pub trait BxDF {
    fn as_any(&self) -> &dyn Any;
    fn matches_flags(&self, t: BxDFType) -> bool {
        self.typ() & t == t
    }
    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum;
    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        u: &Point2f,
        pdf: &mut Float,
        sample_type: Option<&mut BxDFType>,
    ) -> Spectrum {
        *wi = cosine_sample_hemisphere(u);
        if wo.z < 0.0 {
            wi.z *= -1.0;
        }
        *pdf = self.pdf(wo, wi);
        self.f(wo, wi)
    }
    fn rho(&self, w: &Vector3f, n_samples: usize, u: &[Point2f]) -> Spectrum {
        let mut r = Spectrum::new(0.0);
        for i in 0..n_samples {
            let mut wi = Vector3f::default();
            let mut pdf = 0.0;
            let f = self.sample_f(w, &mut wi, &u[i], &mut pdf, None);
            if pdf > 0.0 {
                r += f * abs_cos_theta(&wi) / pdf;
            }
        }
        r / n_samples as Float
    }
    fn rho2(&self, n_samples: usize, samples1: &[Point2f], samples2: &[Point2f]) -> Spectrum {
        let mut r = Spectrum::new(0.0);
        for i in 0..n_samples {
            let mut wo = Vector3f::default();
            let mut wi = Vector3f::default();
            wo = uniform_sample_hemisphere(&samples1[i]);
            let pdfo = uniform_hemisphere_pdf();
            let mut pdfi = 0.0;
            let f = self.sample_f(&wo, &mut wi, &samples2[i], &mut pdfi, None);
            if pdfi > 0.0 {
                r += f * abs_cos_theta(&wi) * abs_cos_theta(&wo) / (pdfo * pdfi);
            }
        }
        r / (PI * n_samples as Float)
    }
    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> Float {
        if same_hemisphere(wo, wi) {
            abs_cos_theta(wi) * INV_PI
        } else {
            0.0
        }
    }
    fn typ(&self) -> BxDFType;
}

pub struct BaseBxDF {
    typ: BxDFType,
}

impl BaseBxDF {
    pub fn new(typ: BxDFType) -> Self {
        Self { typ }
    }
}

pub struct ScaledBxDF {
    base: BaseBxDF,
    bxdf: BxDFDt,
    scale: Spectrum,
}

impl BxDF for ScaledBxDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum {
        self.bxdf.f(wo, wi) * self.scale
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        sample: &Point2f,
        pdf: &mut f32,
        sample_type: Option<&mut BxDFType>,
    ) -> Spectrum {
        let f = self.bxdf.sample_f(wo, wi, sample, pdf, sample_type);
        f * self.scale
    }

    fn rho(&self, wo: &Vector3f, n_samples: usize, samples: &[Point2f]) -> Spectrum {
        self.bxdf.rho(wo, n_samples, samples) * self.scale
    }

    fn rho2(&self, n_samples: usize, samples1: &[Point2f], samples2: &[Point2f]) -> Spectrum {
        self.bxdf.rho2(n_samples, samples1, samples2) * self.scale
    }

    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f32 {
        self.bxdf.pdf(wo, wi)
    }

    fn typ(&self) -> BxDFType {
        self.base.typ
    }
}

pub trait Fresnel {
    fn evaluate(&self, cos_i: Float) -> Spectrum;
}

pub type FresnelDt = Arc<Box<dyn Fresnel>>;

pub struct FresnelConductor {
    eta_i: Spectrum,
    eta_t: Spectrum,
    k: Spectrum,
}

impl FresnelConductor {
    pub fn new(eta_i: Spectrum, eta_t: Spectrum, k: Spectrum) -> Self {
        Self { eta_i, eta_t, k }
    }
}

impl Fresnel for FresnelConductor {
    fn evaluate(&self, cos_i: f32) -> Spectrum {
        fr_conductor(cos_i.abs(), &self.eta_i, &self.eta_t, &self.k)
    }
}

pub struct FresnelDielectric {
    eta_i: Float,
    eta_t: Float,
}

impl FresnelDielectric {
    pub fn new(eta_i: Float, eta_t: Float) -> Self {
        Self { eta_i, eta_t }
    }
}

impl Fresnel for FresnelDielectric {
    fn evaluate(&self, cos_i: f32) -> Spectrum {
        fr_dielectric(cos_i, self.eta_i, self.eta_t).into()
    }
}

pub struct FresnelNoOp;

impl Fresnel for FresnelNoOp {
    fn evaluate(&self, cos_i: f32) -> Spectrum {
        Spectrum::new(1.0)
    }
}

pub struct SpecularReflection {
    base: BaseBxDF,
    r: Spectrum,
    fresnel: FresnelDt,
}

impl SpecularReflection {
    pub fn new(r: Spectrum, fresnel: FresnelDt) -> Self {
        Self {
            base: BaseBxDF::new(BxDFType::BSDF_REFLECTION | BxDFType::BSDF_SPECULAR),
            r,
            fresnel,
        }
    }
}

impl BxDF for SpecularReflection {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum {
        0.0.into()
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        sample: &Point2f,
        pdf: &mut f32,
        sample_type: Option<&mut BxDFType>,
    ) -> Spectrum {
        *wi = Vector3f::new(-wo.x, -wo.y, wo.z);
        *pdf = 1.0;
        self.r * self.fresnel.evaluate(cos_theta(wi)) / abs_cos_theta(wi)
    }

    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f32 {
        0.0
    }

    fn typ(&self) -> BxDFType {
        self.base.typ
    }
}

pub struct SpecularTransmission {
    base: BaseBxDF,
    t: Spectrum,
    eta_a: Float,
    eta_b: Float,
    fresnel: FresnelDielectric,
    mode: TransportMode,
}

impl SpecularTransmission {
    pub fn new(t: Spectrum, eta_a: Float, eta_b: Float, mode: TransportMode) -> Self {
        Self {
            base: BaseBxDF::new(BxDFType::BSDF_SPECULAR | BxDFType::BSDF_TRANSMISSION),
            t,
            eta_a,
            eta_b,
            fresnel: FresnelDielectric::new(eta_a, eta_b),
            mode,
        }
    }
}

impl BxDF for SpecularTransmission {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum {
        0.0.into()
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        sample: &Point2f,
        pdf: &mut f32,
        sample_type: Option<&mut BxDFType>,
    ) -> Spectrum {
        let entering = cos_theta(wo) > 0.0;
        let (eta_i, eta_t) = if entering {
            (self.eta_a, self.eta_b)
        } else {
            (self.eta_b, self.eta_a)
        };

        if !refract(
            wo,
            &Normal3f::new(0.0, 0.0, 1.0).face_forward(*wo),
            eta_i / eta_t,
            wi,
        ) {
            return 0.0.into();
        }
        *pdf = 1.0;
        let mut ft = self.t * (Spectrum::new(1.0) - self.fresnel.evaluate(cos_theta(wi)));
        if let TransportMode::Radiance = self.mode {
            ft *= (eta_i * eta_i) / (eta_t * eta_t);
        }

        ft / abs_cos_theta(wi)
    }

    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f32 {
        0.0
    }

    fn typ(&self) -> BxDFType {
        self.base.typ
    }
}

pub struct FresnelSpecular {
    base: BaseBxDF,
    r: Spectrum,
    t: Spectrum,
    eta_a: Float,
    eta_b: Float,
    mode: TransportMode,
}

impl FresnelSpecular {
    pub fn new(r: Spectrum, t: Spectrum, eta_a: Float, eta_b: Float, mode: TransportMode) -> Self {
        Self {
            base: BaseBxDF::new(
                BxDFType::BSDF_TRANSMISSION | BxDFType::BSDF_SPECULAR | BxDFType::BSDF_REFLECTION,
            ),
            r,
            t,
            eta_a,
            eta_b,
            mode,
        }
    }
}

impl BxDF for FresnelSpecular {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum {
        0.0.into()
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        u: &Point2f,
        pdf: &mut f32,
        sample_type: Option<&mut BxDFType>,
    ) -> Spectrum {
        let f = fr_dielectric(cos_theta(wo), self.eta_a, self.eta_b);
        if u[0] < f {
            *wi = Vector3f::new(-wo.x, -wo.y, wo.z);
            if let Some(sample_type) = sample_type {
                *sample_type = BxDFType::BSDF_SPECULAR | BxDFType::BSDF_REFLECTION;
            }
            *pdf = f;
            self.r * f / abs_cos_theta(wi)
        } else {
            let entering = cos_theta(wo) > 0.0;
            let (eta_i, eta_t) = if entering {
                (self.eta_a, self.eta_b)
            } else {
                (self.eta_b, self.eta_a)
            };

            if !refract(
                wo,
                &Normal3f::new(0.0, 0.0, 1.0).face_forward(*wo),
                eta_i / eta_t,
                wi,
            ) {
                return 0.0.into();
            }

            let mut ft = self.t * (1.0 - f);

            if let TransportMode::Radiance = self.mode {
                ft *= (eta_i * eta_i) / (eta_t * eta_t);
            }
            if let Some(sample_type) = sample_type {
                *sample_type = BxDFType::BSDF_SPECULAR | BxDFType::BSDF_TRANSMISSION;
            }
            *pdf = 1.0 - f;
            ft / abs_cos_theta(wi)
        }
    }

    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f32 {
        0.0
    }

    fn typ(&self) -> BxDFType {
        self.base.typ
    }
}

pub struct LambertianReflection {
    base: BaseBxDF,
    r: Spectrum,
}

impl LambertianReflection {
    pub fn new(r: Spectrum) -> Self {
        Self {
            base: BaseBxDF::new(BxDFType::BSDF_REFLECTION | BxDFType::BSDF_DIFFUSE),
            r,
        }
    }
}

impl BxDF for LambertianReflection {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum {
        self.r * INV_PI
    }

    fn rho(&self, wo: &Vector3f, n_samples: usize, samples: &[Point2f]) -> Spectrum {
        self.r
    }

    fn rho2(&self, n_samples: usize, samples1: &[Point2f], samples2: &[Point2f]) -> Spectrum {
        self.r
    }

    fn typ(&self) -> BxDFType {
        self.base.typ
    }
}

pub struct LambertianTransmission {
    base: BaseBxDF,
    t: Spectrum,
}

impl LambertianTransmission {
    pub fn new(t: Spectrum) -> Self {
        Self {
            base: BaseBxDF::new(BxDFType::BSDF_TRANSMISSION | BxDFType::BSDF_DIFFUSE),
            t,
        }
    }
}

impl BxDF for LambertianTransmission {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum {
        self.t * INV_PI
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        u: &Point2f,
        pdf: &mut f32,
        sample_type: Option<&mut BxDFType>,
    ) -> Spectrum {
        *wi = cosine_sample_hemisphere(u);
        if wo.z > 0.0 {
            wi.z *= -1.0;
        }
        *pdf = self.pdf(wo, wi);
        self.f(wo, wi)
    }

    fn rho(&self, wo: &Vector3f, n_samples: usize, samples: &[Point2f]) -> Spectrum {
        self.t
    }

    fn rho2(&self, n_samples: usize, samples1: &[Point2f], samples2: &[Point2f]) -> Spectrum {
        self.t
    }

    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f32 {
        if !same_hemisphere(wo, wi) {
            abs_cos_theta(wi) * INV_PI
        } else {
            0.0
        }
    }

    fn typ(&self) -> BxDFType {
        self.base.typ
    }
}

pub struct OrenNayar {
    base: BaseBxDF,
    r: Spectrum,
    a: Float,
    b: Float,
}

impl OrenNayar {
    pub fn new(r: Spectrum, sigma: Float) -> Self {
        let signma = radians(sigma);
        let sigma2 = sigma * sigma;
        let a = 1.0 - (sigma2 / (2.0 * (sigma2 + 0.33)));
        let b = 0.45 * sigma2 / (sigma2 + 0.09);
        Self {
            base: BaseBxDF::new(BxDFType::BSDF_REFLECTION | BxDFType::BSDF_DIFFUSE),
            a,
            b,
            r,
        }
    }
}

impl BxDF for OrenNayar {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum {
        let sin_theta_i = sin_theta(wi);
        let sin_theta_o = sin_theta(wo);

        let mut max_cos = 0.0;
        if sin_theta_i > 1e-4 && sin_theta_o > 1e-4 {
            let sin_phi_i = sin_phi(wi);
            let cos_phi_i = cos_phi(wi);
            let sin_phi_o = sin_theta(wo);
            let cos_phi_o = cos_phi(wo);
            let d_cos = cos_phi_i * cos_phi_o + sin_theta_i * sin_theta_o;
            max_cos = d_cos.max(0.0);
        }

        let mut sin_alpha = 0.0;
        let mut tan_beta = 0.0;

        if abs_cos_theta(wi) > abs_cos_theta(wo) {
            sin_alpha = sin_theta_o;
            tan_beta = sin_theta_i / abs_cos_theta(wi);
        } else {
            sin_alpha = sin_theta_i;
            tan_beta = sin_theta_o / abs_cos_theta(wo);
        }

        self.r * INV_PI * (self.a + self.b * max_cos * sin_alpha * tan_beta)
    }

    fn typ(&self) -> BxDFType {
        self.base.typ
    }
}

pub struct MicrofacetReflection {
    base: BaseBxDF,
    r: Spectrum,
    distribution: MicrofacetDistributionDt,
    fresnel: FresnelDt,
}

impl MicrofacetReflection {
    pub fn new(r: Spectrum, distribution: MicrofacetDistributionDt, fresnel: FresnelDt) -> Self {
        Self {
            r,
            distribution,
            fresnel,
            base: BaseBxDF::new(BxDFType::BSDF_GLOSSY | BxDFType::BSDF_REFLECTION),
        }
    }
}

impl BxDF for MicrofacetReflection {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum {
        let cos_theta_o = abs_cos_theta(wo);
        let cos_theta_i = abs_cos_theta(wi);
        let mut wh = *wi + *wo;

        if cos_theta_i == 0.0 || cos_theta_o == 0.0 {
            return 0.0.into();
        }
        if wh.x == 0.0 && wh.y == 0.0 && wh.z == 0.0 {
            return 0.0.into();
        }

        wh = wh.normalize();
        let f = self
            .fresnel
            .evaluate(wi.dot(&wh.face_forward(Vector3f::new(0.0, 0.0, 1.0))));
        self.r * self.distribution.d(&wh) * self.distribution.g(wo, wi) * f
            / (4.0 * cos_theta_i * cos_theta_o)
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        u: &Point2f,
        pdf: &mut f32,
        sample_type: Option<&mut BxDFType>,
    ) -> Spectrum {
        if wo.z == 0.0 {
            return 0.0.into();
        }

        let wh = self.distribution.sample_wh(wo, u);
        if wo.dot(&wh) < 0.0 {
            return 0.0.into();
        }

        if !same_hemisphere(wo, wi) {
            return 0.0.into();
        }

        *pdf = self.distribution.pdf(wo, &wh) / (4.0 * wo.dot(&wh));
        self.f(wo, wi)
    }

    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f32 {
        if !same_hemisphere(wo, wi) {
            return 0.0.into();
        }
        let wh = (*wo + *wi).normalize();
        self.distribution.pdf(wo, &wh) / (4.0 * wo.dot(&wh))
    }

    fn typ(&self) -> BxDFType {
        self.base.typ
    }
}

pub struct MicrofacetTransmission {
    base: BaseBxDF,
    t: Spectrum,
    distribution: MicrofacetDistributionDt,
    eta_a: Float,
    eta_b: Float,
    fresnel: FresnelDielectric,
    mode: TransportMode,
}

impl MicrofacetTransmission {
    pub fn new(
        t: Spectrum,
        distribution: MicrofacetDistributionDt,
        eta_a: Float,
        eta_b: Float,
        mode: TransportMode,
    ) -> Self {
        Self {
            base: BaseBxDF::new(BxDFType::BSDF_TRANSMISSION | BxDFType::BSDF_GLOSSY),
            t,
            distribution,
            eta_a,
            eta_b,
            mode,
            fresnel: FresnelDielectric::new(eta_a, eta_b),
        }
    }
}

impl BxDF for MicrofacetTransmission {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum {
        if same_hemisphere(wo, wi) {
            return 0.0.into();
        }

        let cos_theta_o = cos_theta(wo);
        let cos_theta_i = cos_theta(wi);
        if cos_theta_i == 0.0 || cos_theta_o == 0.0 {
            return 0.0.into();
        }

        let eta = if cos_theta(wo) > 0.0 {
            self.eta_b / self.eta_a
        } else {
            self.eta_a / self.eta_b
        };

        let mut wh = (*wo + *wi * eta).normalize();
        if wh.z < 0.0 {
            wh = -wh;
        }

        if wo.dot(&wh) * wi.dot(&wh) > 0.0 {
            return 0.0.into();
        }

        let f = self.fresnel.evaluate(wo.dot(&wh));
        let sqrt_denom = wo.dot(&wh) + eta * wi.dot(&wh);
        let factor = if let TransportMode::Radiance = self.mode {
            1.0 / eta
        } else {
            1.0
        };
        (Spectrum::new(1.0) - f)
            * self.t
            * ((self.distribution.d(&wh)
                * self.distribution.g(wo, wi)
                * eta
                * eta
                * wi.abs_dot(&wh)
                * wo.abs_dot(&wh)
                * factor
                * factor)
                / (cos_theta_i * cos_theta_o * sqrt_denom * sqrt_denom))
                .abs()
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        u: &Point2f,
        pdf: &mut f32,
        sample_type: Option<&mut BxDFType>,
    ) -> Spectrum {
        if wo.z == 0.0 {
            return 0.0.into();
        }

        let wh = self.distribution.sample_wh(wo, u);
        if wo.dot(&wh) < 0.0 {
            return 0.0.into();
        }

        let eta = if cos_theta(wo) > 0.0 {
            self.eta_a / self.eta_b
        } else {
            self.eta_b / self.eta_a
        };

        if !refract(wo, &wh, eta, wi) {
            return 0.0.into();
        }

        *pdf = self.pdf(wo, wi);
        self.f(wo, wi)
    }

    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f32 {
        if same_hemisphere(wo, wi) {
            return 0.0.into();
        }
        let eta = if cos_theta(wo) > 0.0 {
            self.eta_b / self.eta_a
        } else {
            self.eta_a / self.eta_b
        };
        let wh = (*wo + *wi * eta).normalize();
        if wo.dot(&wh) * wi.dot(&wh) > 0.0 {
            return 0.0.into();
        }
        let sqrt_denom = wo.dot(&wh) + eta * wi.dot(&wh);
        let dwh_dwi = ((eta * eta * wi.dot(&wh)) / sqrt_denom * sqrt_denom).abs();
        self.distribution.pdf(wo, &wh) * dwh_dwi
    }

    fn typ(&self) -> BxDFType {
        self.base.typ
    }
}

pub struct FresnelBlend {
    base: BaseBxDF,
    rd: Spectrum,
    rs: Spectrum,
    distribution: MicrofacetDistributionDt,
}

impl FresnelBlend {
    pub fn new(rd: Spectrum, rs: Spectrum, distribution: MicrofacetDistributionDt) -> Self {
        Self {
            base: BaseBxDF::new(BxDFType::BSDF_GLOSSY | BxDFType::BSDF_REFLECTION),
            rd,
            rs,
            distribution,
        }
    }

    pub fn schlick_fresnel(&self, cos_theta: Float) -> Spectrum {
        let pow5 = |v| (v * v) * (v * v) * v;
        self.rs + (Spectrum::new(1.0) - self.rs) * pow5(1.0 - cos_theta)
    }
}

impl BxDF for FresnelBlend {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum {
        let pow5 = |v: Float| -> Float { (v * v) * (v * v) * v };
        let diffuse = Spectrum::new(28.0 / (23.0 * PI))
            * self.rd
            * (Spectrum::new(1.0) - self.rs)
            * (1.0 - pow5(1.0 - 0.5 * abs_cos_theta(wi)))
            * (1.0 - pow5(1.0 - 0.5 * abs_cos_theta(wo)));
        let wh = *wi + *wo;
        if wh.x == 0.0 && wh.y == 0.0 && wh.z == 0.0 {
            return 0.0.into();
        }
        let wh = wh.normalize();
        let specular = self.schlick_fresnel(wi.dot(&wh))
            * (self.distribution.d(&wh)
                / (4.0 * wi.abs_dot(&wh) * abs_cos_theta(wi).max(abs_cos_theta(wo))));
        diffuse + specular
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        u_orig: &Point2f,
        pdf: &mut f32,
        sample_type: Option<&mut BxDFType>,
    ) -> Spectrum {
        let mut u = *u_orig;
        if u[0] < 0.5 {
            u[0] = ONE_MINUS_EPSILON.min(2.0 * u[0]);
            *wi = cosine_sample_hemisphere(&u);
            if wo.z < 0.0 {
                wi.z *= -1.0;
            }
        } else {
            u[0] = ONE_MINUS_EPSILON.min(2.0 * (u[0] - 0.5));
            let wh = self.distribution.sample_wh(wo, &u);
            *wi = reflect(wo, &wh);
            if !same_hemisphere(wo, wi) {
                return 0.0.into();
            }
        }
        *pdf = self.pdf(wo, wi);
        self.f(wo, wi)
    }

    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f32 {
        if !same_hemisphere(wo, wi) {
            return 0.0;
        }

        let wh = (*wo + *wi).normalize();
        let pdf_wh = self.distribution.pdf(wo, &wh);
        0.5 * (abs_cos_theta(wi) * INV_PI + pdf_wh / (4.0 * wo.dot(&wh)))
    }

    fn typ(&self) -> BxDFType {
        self.base.typ
    }
}

pub struct FourierBSDF {
    base: BaseBxDF,
    bsdf_table: FourierBSDFTable,
    mode: TransportMode,
}

impl FourierBSDF {
    pub fn new(bsdf_table: FourierBSDFTable, mode: TransportMode) -> Self {
        Self {
            base: BaseBxDF::new(
                BxDFType::BSDF_REFLECTION | BxDFType::BSDF_TRANSMISSION | BxDFType::BSDF_GLOSSY,
            ),
            bsdf_table,
            mode,
        }
    }
}

impl BxDF for FourierBSDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum {
        let mu_i = cos_theta(&-*wi);
        let mu_o = cos_theta(wo);
        let cos_phi = cos_d_phi(-*wi, *wo);

        let mut offset_i = 0;
        let mut offset_o = 0;

        let mut weights_i = [0.0; 4];
        let mut weights_o = [0.0; 4];
        if !self
            .bsdf_table
            .get_weights_and_offset(mu_i, &mut offset_i, &mut weights_i)
            || !self
                .bsdf_table
                .get_weights_and_offset(mu_o, &mut offset_o, &mut weights_o)
        {
            return 0.0.into();
        }

        let mut ak = vec![0.0; self.bsdf_table.m_max * self.bsdf_table.n_channels];

        let mut m_max = 0;
        for b in 0..4 {
            for a in 0..4 {
                let weight = weights_i[a] * weights_o[b];
                if weight != 0.0 {
                    let mut m = 0;
                    let ap = self.bsdf_table.get_ak(offset_i + a, offset_o + b, &mut m);
                    m_max = std::cmp::max(m_max, m);
                    for c in 0..self.bsdf_table.n_channels {
                        for k in 0..m {
                            ak[c * self.bsdf_table.m_max + k] += weight * ap[c * m + k];
                        }
                    }
                }
            }
        }

        let y = fourier(ak.as_slice(), m_max, cos_phi as f64).max(0.0);
        let mut scale = if mu_i != 0.0 { 1.0 / mu_i.abs() } else { 0.0 };

        if let TransportMode::Radiance = self.mode {
            if mu_i * mu_o > 0.0 {
                let eta = if mu_i > 0.0 {
                    1.0 / self.bsdf_table.eta
                } else {
                    self.bsdf_table.eta
                };
                scale *= eta * eta;
            }
        }
        if self.bsdf_table.n_channels == 1 {
            (y * scale).into()
        } else {
            let r = fourier(
                &ak.as_slice()[1 * self.bsdf_table.m_max..],
                m_max,
                cos_phi as f64,
            );
            let b = fourier(
                &ak.as_slice()[2 * self.bsdf_table.m_max..],
                m_max,
                cos_phi as f64,
            );
            let g = 1.39829 * y - 0.100913 * b - 0.297375 * r;
            let rgb = [r * scale, g * scale, b * scale];
            Spectrum::from_rgb(&rgb, SpectrumType::Reflectance).clamp(0.0, Float::INFINITY)
        }
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        u: &Point2f,
        pdf: &mut f32,
        sample_type: Option<&mut BxDFType>,
    ) -> Spectrum {
        let mu_o = cos_theta(wo);
        let mut pdf_mu = 0.0;
        let mu_i = sample_catmull_rom_2d(
            self.bsdf_table.n_mu,
            self.bsdf_table.n_mu,
            self.bsdf_table.mu.as_slice(),
            self.bsdf_table.mu.as_slice(),
            self.bsdf_table.a0.as_slice(),
            self.bsdf_table.cdf.as_slice(),
            mu_o,
            u[1],
            None,
            Some(&mut pdf_mu),
        );

        let mut offset_i = 0;
        let mut offset_o = 0;
        let mut weights_i = [0.0; 4];
        let mut weights_o = [0.0; 4];

        let mut ak = vec![0.0; self.bsdf_table.m_max * self.bsdf_table.n_channels];

        let mut m_max = 0;
        for b in 0..4 {
            for a in 0..4 {
                let weight = weights_i[a] * weights_o[b];
                if weight != 0.0 {
                    let mut m = 0;
                    let ap = self.bsdf_table.get_ak(offset_i + a, offset_o + b, &mut m);
                    m_max = std::cmp::max(m_max, m);
                    for c in 0..self.bsdf_table.n_channels {
                        for k in 0..m {
                            ak[c * self.bsdf_table.m_max + k] += weight * ap[c * m + k];
                        }
                    }
                }
            }
        }

        let mut phi = 0.0;
        let mut pdf_phi = 0.0;
        let y = sample_fourier(
            ak.as_slice(),
            self.bsdf_table.recip.as_slice(),
            m_max,
            u[0],
            &mut pdf_phi,
            &mut phi,
        );
        *pdf = (pdf_phi * pdf_mu).max(0.0);

        let sin2_theta_i = (1.0 - mu_i * mu_i).max(0.0);
        let mut norm = (sin2_theta_i / sin2_theta(wo)).sqrt();
        if norm.is_infinite() {
            norm = 0.0;
        }
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        *wi = -Vector3f::new(
            norm * (cos_phi * wo.x - sin_phi * wo.y),
            norm * (sin_phi * wo.x + cos_phi * wo.y),
            mu_i,
        );
        *wi = wi.normalize();
        let mut scale = if mu_i != 0.0 { 1.0 / mu_i.abs() } else { 0.0 };
        if let TransportMode::Radiance = self.mode {
            if mu_i * mu_o > 0.0 {
                let eta = if mu_i > 0.0 {
                    1.0 / self.bsdf_table.eta
                } else {
                    self.bsdf_table.eta
                };
                scale *= eta * eta;
            }
        }

        if self.bsdf_table.n_channels == 1 {
            return (y * scale).into();
        }
        let r = fourier(
            &ak.as_slice()[1 * self.bsdf_table.m_max..],
            m_max,
            cos_phi as f64,
        );
        let b = fourier(
            &ak.as_slice()[2 * self.bsdf_table.m_max..],
            m_max,
            cos_phi as f64,
        );
        let g = 1.39829 * y - 0.100913 * b - 0.297375 * r;
        let rgb = [r * scale, g * scale, b * scale];
        Spectrum::from_rgb(&rgb, SpectrumType::Reflectance).clamp(0.0, Float::INFINITY)
    }

    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f32 {
        let mu_i = cos_theta(&-*wi);
        let mu_o = cos_theta(wo);
        let cos_phi = cos_d_phi(-*wi, *wo);
        let mut offset_i = 0;
        let mut offset_o = 0;
        let mut weights_i = [0.0; 4];
        let mut weights_o = [0.0; 4];
        if !self
            .bsdf_table
            .get_weights_and_offset(mu_i, &mut offset_i, &mut weights_i)
            || !self
                .bsdf_table
                .get_weights_and_offset(mu_o, &mut offset_o, &mut weights_o)
        {
            return 0.0;
        }
        let mut ak = vec![0.0; self.bsdf_table.m_max];
        let mut m_max = 0;
        for o in 0..4 {
            for i in 0..4 {
                let weight = weights_i[i] * weights_o[o];
                if weight == 0.0 {
                    continue;
                }

                let mut order = 0;
                let coeffs = self
                    .bsdf_table
                    .get_ak(offset_i + i, offset_o + o, &mut order);
                m_max = std::cmp::max(m_max, order);
                for k in 0..order {
                    ak[k] += coeffs[k] * weight;
                }
            }
        }
        let mut rho = 0.0;
        for o in 0..4 {
            if weights_o[o] == 0.0 {
                continue;
            }
            rho += weights_o[o]
                * self.bsdf_table.cdf
                    [(offset_o + o) * self.bsdf_table.n_mu + self.bsdf_table.n_mu - 1]
                * (2.0 * PI);
        }
        let y = fourier(ak.as_slice(), m_max, cos_phi as f64);
        if rho > 0.0 && y > 0.0 {
            y / rho
        } else {
            0.0
        }
    }

    fn typ(&self) -> BxDFType {
        self.base.typ
    }
}
