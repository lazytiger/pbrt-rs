use crate::core::{
    geometry::{Normal3f, Point2f, Vector3f},
    interaction::SurfaceInteraction,
    material::{MaterialDt, TransportMode},
    pbrt::{Float, PI},
    reflection::{cos_theta, fr_dielectric, BaseBxDF, BxDF, BxDFType},
    scene::Scene,
    spectrum::Spectrum,
};
use derive_more::{Deref, DerefMut};
use std::{
    any::Any,
    sync::{Arc, Mutex, RwLock},
};

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
    fn eta(&self) -> Float;
}

pub struct BaseBSSRDF<'a> {
    po: &'a SurfaceInteraction,
    eta: Float,
}

impl<'a> BaseBSSRDF<'a> {
    pub fn new(po: &'a SurfaceInteraction, eta: Float) -> Self {
        Self { po, eta }
    }
}

pub trait SeparableBSSRDF: BSSRDF {
    fn sr(&self, d: Float) -> Spectrum;
    fn sampler_sr(&self, ch: usize, u: Float) -> Float;
    fn pdf_sr(&self, ch: usize, r: Float) -> Float;

    fn sw(&self, w: &Vector3f) -> Spectrum {
        let c = 1.0 - 2.0 * fresnel_moment1(1.0 / self.eta());
        ((1.0 - fr_dielectric(cos_theta(w), 1.0, self.eta())) / (c * PI)).into()
    }
    fn mode(&self) -> TransportMode;
}

#[derive(Deref, DerefMut)]
pub struct BaseSeparableBSSRDF<'a> {
    #[deref]
    #[deref_mut]
    base: BaseBSSRDF<'a>,
    ns: Normal3f,
    ss: Vector3f,
    ts: Vector3f,
    material: MaterialDt,
    mode: TransportMode,
}

impl<'a> BaseSeparableBSSRDF<'a> {
    pub fn new(
        po: &'a SurfaceInteraction,
        eta: Float,
        material: MaterialDt,
        mode: TransportMode,
    ) -> Self {
        let ns = po.shading.n;
        let ss = po.shading.dpdu.normalize();
        let ts = ns.cross(&ns);
        Self {
            base: BaseBSSRDF::new(po, eta),
            ns,
            ss,
            ts,
            material,
            mode,
        }
    }

    pub fn sp(&self, pi: &SurfaceInteraction) -> Spectrum {
        self.sr(self.po.p.distance(&pi.p))
    }

    pub fn sample_s(
        &self,
        scene: &Scene,
        u1: Float,
        u2: &Point2f,
        si: &mut SurfaceInteraction,
        pdf: &mut Float,
    ) -> Spectrum {
        todo!()
    }

    pub fn sample_sp(
        &self,
        scene: &Scene,
        u1: Float,
        u2: Point2f,
        si: &mut SurfaceInteraction,
        pdf: &mut Float,
    ) -> Spectrum {
        todo!()
    }

    pub fn pdf_sp(&self, si: &SurfaceInteraction) -> Float {
        todo!()
    }
}

impl<'a> BSSRDF for BaseSeparableBSSRDF<'a> {
    fn s(&self, pi: &SurfaceInteraction, wi: &Vector3f) -> Spectrum {
        let ft = fr_dielectric(cos_theta(&self.po.wo), 1.0, self.eta);
        self.sp(pi) * self.sw(wi) * (1.0 - ft)
    }

    fn sample_s(
        &self,
        scene: &Scene,
        u1: f32,
        u2: &Point2f,
        si: &mut SurfaceInteraction,
        pdf: &mut f32,
    ) -> Spectrum {
        todo!()
    }

    fn eta(&self) -> f32 {
        self.eta
    }
}

impl<'a> SeparableBSSRDF for BaseSeparableBSSRDF<'a> {
    fn sr(&self, d: f32) -> Spectrum {
        unimplemented!()
    }

    fn sampler_sr(&self, ch: usize, u: f32) -> f32 {
        unimplemented!()
    }

    fn pdf_sr(&self, ch: usize, r: f32) -> f32 {
        unimplemented!()
    }

    fn mode(&self) -> TransportMode {
        self.mode
    }
}

#[derive(Deref, DerefMut)]
pub struct TabulateBSSRDF<'a, 'b> {
    #[deref]
    #[deref_mut]
    base: BaseSeparableBSSRDF<'b>,
    table: &'a BSSRDFTable,
    sigma_t: Spectrum,
    rho: Spectrum,
}

impl<'a, 'b> BSSRDF for TabulateBSSRDF<'a, 'b> {
    fn s(&self, pi: &SurfaceInteraction, wi: &Vector3f) -> Spectrum {
        self.base.s(pi, wi)
    }

    fn sample_s(
        &self,
        scene: &Scene,
        u1: f32,
        u2: &Point2f,
        si: &mut SurfaceInteraction,
        pdf: &mut f32,
    ) -> Spectrum {
        self.base.sample_s(scene, u1, u2, si, pdf)
    }

    fn eta(&self) -> f32 {
        self.eta
    }
}

impl<'a, 'b> SeparableBSSRDF for TabulateBSSRDF<'a, 'b> {
    fn sr(&self, d: f32) -> Spectrum {
        todo!()
    }

    fn sampler_sr(&self, ch: usize, u: f32) -> f32 {
        todo!()
    }

    fn pdf_sr(&self, ch: usize, r: f32) -> f32 {
        todo!()
    }

    fn mode(&self) -> TransportMode {
        self.mode
    }
}

pub struct BSSRDFTable {
    n_rho_samples: usize,
    n_radius_samples: usize,
    rho_samples: Vec<Float>,
    radius_samples: Vec<Float>,
    profile: Vec<Float>,
    rho_eff: Vec<Float>,
    profile_cdf: Vec<Float>,
}

impl BSSRDFTable {
    pub fn new(n_rho_samples: usize, n_radius_samples: usize) -> Self {
        todo!()
    }

    pub fn eval_profile(&self, rho_index: usize, radius_index: usize) -> Float {
        self.profile[rho_index * self.n_radius_samples + radius_index]
    }
}

#[derive(Deref, DerefMut)]
pub struct SeparableBSSRDFAdapter {
    #[deref]
    #[deref_mut]
    base: BaseBxDF,
    bssrdf: SeparableBSSRDFDt,
}

pub type SeparableBSSRDFDt = Arc<Box<dyn SeparableBSSRDF>>;

impl BxDF for SeparableBSSRDFAdapter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn f(&self, wo: &Vector3f, wi: &Vector3f) -> Spectrum {
        let mut f = self.bssrdf.sw(wi);
        if let TransportMode::Radiance = self.bssrdf.mode() {
            f *= self.bssrdf.eta() * self.bssrdf.eta();
        }
        f
    }

    fn typ(&self) -> BxDFType {
        self.typ
    }
}

pub fn beam_diffusion_ss(sigma_s: Float, sigma_a: Float, g: Float, eta: Float, r: Float) -> Float {
    todo!()
}

pub fn beam_diffusion_ms(sigma_s: Float, sigma_a: Float, g: Float, eta: Float, r: Float) -> Float {
    todo!()
}

pub fn compute_beam_diffusion_bssrdf(g: Float, eta: Float, t: &mut BSSRDFTable) {
    todo!()
}

pub fn subsurface_from_diffuse(
    table: &BSSRDFTable,
    rho_eff: &Spectrum,
    mfp: &Spectrum,
    sigma_a: &mut Spectrum,
    sigma_s: &mut Spectrum,
) {
    todo!()
}
