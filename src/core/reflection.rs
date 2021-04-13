use crate::core::{
    geometry::{Normal3f, Point2f, Vector3f},
    interaction::SurfaceInteraction,
    material::TransportMode,
    microfacet::MicrofacetDistributionDt,
    pbrt::{clamp, radians, Float},
    spectrum::Spectrum,
};
use bitflags::bitflags;
use std::{any::Any, sync::Arc};

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

    pub fn f(&self, wo: &Vector3f, wi: &Vector3f, flags: BxDFType) -> Spectrum {
        todo!()
    }

    pub fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        sample: &Point2f,
        pdf: &mut f32,
        typ: BxDFType,
        sample_type: &mut BxDFType,
    ) -> Spectrum {
        todo!()
    }

    pub fn rho(
        &self,
        wo: &Vector3f,
        n_samples: usize,
        samples: &[Point2f],
        flags: BxDFType,
    ) -> Spectrum {
        todo!()
    }

    pub fn rho2(
        &self,
        n_samples: usize,
        samples1: &[Point2f],
        samples2: &[Point2f],
        flags: BxDFType,
    ) -> Spectrum {
        todo!()
    }

    pub fn pdf(&self, wo: &Vector3f, wi: &Vector3f, flags: BxDFType) -> f32 {
        todo!()
    }
}

pub type BxDFDt = Arc<Box<dyn BxDF>>;

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
        sample: &Point2f,
        pdf: &mut Float,
        sample_type: &mut BxDFType,
    ) -> Spectrum {
        todo!()
    }
    fn rho(&self, wo: &Vector3f, n_samples: usize, samples: &[Point2f]) -> Spectrum {
        todo!()
    }
    fn rho2(&self, n_samples: usize, samples1: &[Point2f], samples2: &[Point2f]) -> Spectrum {
        todo!()
    }
    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> Float {
        todo!()
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
        todo!()
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        sample: &Point2f,
        pdf: &mut f32,
        sample_type: &mut BxDFType,
    ) -> Spectrum {
        todo!()
    }

    fn rho(&self, wo: &Vector3f, n_samples: usize, samples: &[Point2f]) -> Spectrum {
        todo!()
    }

    fn rho2(&self, n_samples: usize, samples1: &[Point2f], samples2: &[Point2f]) -> Spectrum {
        todo!()
    }

    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f32 {
        todo!()
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
        todo!()
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
        todo!()
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
        todo!()
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        sample: &Point2f,
        pdf: &mut f32,
        sample_type: &mut BxDFType,
    ) -> Spectrum {
        todo!()
    }

    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f32 {
        1.0
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
        todo!()
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        sample: &Point2f,
        pdf: &mut f32,
        sample_type: &mut BxDFType,
    ) -> Spectrum {
        todo!()
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
        todo!()
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        sample: &Point2f,
        pdf: &mut f32,
        sample_type: &mut BxDFType,
    ) -> Spectrum {
        todo!()
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
        todo!()
    }

    fn rho(&self, wo: &Vector3f, n_samples: usize, samples: &[Point2f]) -> Spectrum {
        todo!()
    }

    fn rho2(&self, n_samples: usize, samples1: &[Point2f], samples2: &[Point2f]) -> Spectrum {
        todo!()
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
        todo!()
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        sample: &Point2f,
        pdf: &mut f32,
        sample_type: &mut BxDFType,
    ) -> Spectrum {
        todo!()
    }

    fn rho(&self, wo: &Vector3f, n_samples: usize, samples: &[Point2f]) -> Spectrum {
        todo!()
    }

    fn rho2(&self, n_samples: usize, samples1: &[Point2f], samples2: &[Point2f]) -> Spectrum {
        todo!()
    }

    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f32 {
        todo!()
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
        todo!()
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
        todo!()
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        sample: &Point2f,
        pdf: &mut f32,
        sample_type: &mut BxDFType,
    ) -> Spectrum {
        todo!()
    }

    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f32 {
        todo!()
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
        todo!()
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        sample: &Point2f,
        pdf: &mut f32,
        sample_type: &mut BxDFType,
    ) -> Spectrum {
        todo!()
    }

    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f32 {
        todo!()
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
        todo!()
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        sample: &Point2f,
        pdf: &mut f32,
        sample_type: &mut BxDFType,
    ) -> Spectrum {
        todo!()
    }

    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f32 {
        todo!()
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
        todo!()
    }

    fn sample_f(
        &self,
        wo: &Vector3f,
        wi: &mut Vector3f,
        sample: &Point2f,
        pdf: &mut f32,
        sample_type: &mut BxDFType,
    ) -> Spectrum {
        todo!()
    }

    fn pdf(&self, wo: &Vector3f, wi: &Vector3f) -> f32 {
        todo!()
    }

    fn typ(&self) -> BxDFType {
        self.base.typ
    }
}
