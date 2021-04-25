use crate::{
    core::{
        geometry::{Normal3f, Point2f, Vector3f},
        interaction::{BaseInteraction, SurfaceInteraction},
        interpolation::{
            catmull_rom_weights, integrate_catmull_rom, invert_catmull_rom, sample_catmull_rom_2d,
        },
        material::{MaterialDt, TransportMode},
        medium::phase_hg,
        pbrt::{any_equal, clamp, Float, INV_4_PI, PI},
        reflection::{cos_theta, fr_dielectric, BaseBxDF, BxDF, BxDFType, BSDF},
        scene::Scene,
        spectrum::Spectrum,
    },
    integrators::bdpt::VertexType::Surface,
    parallel_for,
};
use derive_more::{Deref, DerefMut};
use num::traits::clamp_max;
use std::{
    any::Any,
    sync::{Arc, Mutex, RwLock},
};

pub fn fresnel_moment1(eta: Float) -> Float {
    let eta2 = eta * eta;
    let eta3 = eta2 * eta;
    let eta4 = eta3 * eta;
    let eta5 = eta4 * eta;
    if eta < 1.0 {
        0.45966 - 1.73965 * eta + 3.37668 * eta2 - 3.904945 * eta3 + 2.49277 * eta4 - 0.68441 * eta5
    } else {
        -4.61686 + 11.1136 * eta - 10.4646 * eta2 + 5.11455 * eta3 - 1.27198 * eta4 + 0.12746 * eta5
    }
}

pub fn fresnel_moment2(eta: Float) -> Float {
    let eta2 = eta * eta;
    let eta3 = eta2 * eta;
    let eta4 = eta3 * eta;
    let eta5 = eta4 * eta;
    if eta < 1.0 {
        0.27614 - 0.87350 * eta + 1.12077 * eta2 - 0.65095 * eta3 + 0.07883 * eta4 + 0.04860 * eta5
    } else {
        let r_eta = 1.0 / eta;
        let r_eta2 = r_eta * r_eta;
        let r_eta3 = r_eta2 * r_eta;
        -547.033 + 45.3087 * r_eta3 - 218.725 * r_eta2 + 458.843 * r_eta + 404.557 * eta
            - 189.519 * eta2
            + 54.9327 * eta3
            - 9.00603 * eta4
            + 0.63942 * eta5
    }
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

#[derive(Clone)]
pub struct BaseBSSRDF {
    po: SurfaceInteraction,
    eta: Float,
}

impl BaseBSSRDF {
    pub fn new(po: SurfaceInteraction, eta: Float) -> Self {
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

#[derive(Deref, DerefMut, Clone)]
pub struct BaseSeparableBSSRDF {
    #[deref]
    #[deref_mut]
    base: BaseBSSRDF,
    ns: Normal3f,
    ss: Vector3f,
    ts: Vector3f,
    material: MaterialDt,
    mode: TransportMode,
}

impl BaseSeparableBSSRDF {
    pub fn new(
        po: SurfaceInteraction,
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

    pub fn sample_sp(
        &self,
        scene: &Scene,
        mut u1: Float,
        u2: &Point2f,
        si: &mut SurfaceInteraction,
        pdf: &mut Float,
    ) -> Spectrum {
        let mut vx = Vector3f::default();
        let mut vy = Vector3f::default();
        let mut vz = Vector3f::default();
        if u1 < 0.5 {
            vx = self.ss;
            vy = self.ts;
            vz = self.ns;
            u1 *= 2.0;
        } else if u1 < 0.75 {
            vx = self.ts;
            vy = self.ns;
            vz = self.ss;
            u1 = (u1 - 0.5) * 4.0;
        } else {
            vx = self.ns;
            vy = self.ss;
            vz = self.ts;
            u1 = (u1 - 0.75) * 4.0;
        }

        let ch = clamp(
            (u1 * Spectrum::n_samples() as Float) as isize,
            0,
            Spectrum::n_samples() as isize - 1,
        ) as usize;
        u1 = u1 * Spectrum::n_samples() as Float - ch as Float;

        let r = self.sampler_sr(ch, u2[0]);
        if r < 0.0 {
            return Spectrum::new(0.0);
        }
        let phi = 2.0 * PI * u2[1];

        let r_max = self.sampler_sr(ch, 0.999);
        if r >= r_max {
            return Spectrum::new(0.0);
        }
        let l = 2.0 * (r_max * r_max - r * r).sqrt();

        let mut base = SurfaceInteraction::default();
        base.p = self.po.p + (vx * phi.cos() + vy * phi.sin()) * r - vz * l * 0.5;
        base.time = self.po.time;
        let p_target = base.p + vz * l;
        let mut base = Arc::new(base);

        struct InteractionChain {
            si: Arc<SurfaceInteraction>,
            next: Option<Arc<InteractionChain>>,
        }

        impl InteractionChain {
            fn get_mut_si(&mut self) -> &mut SurfaceInteraction {
                Arc::get_mut(&mut self.si).unwrap()
            }

            pub fn new() -> Self {
                Self {
                    si: Arc::new(SurfaceInteraction::default()),
                    next: None,
                }
            }
        }

        let mut chain = Arc::new(InteractionChain::new());
        let mut ptr = chain.clone();
        let mut n_found = 0;
        loop {
            let mut r = base.spawn_ray(&p_target);
            if r.d == Vector3f::default()
                || !scene.intersect(&mut r, Arc::get_mut(&mut ptr).unwrap().get_mut_si())
            {
                break;
            }
            base = ptr.si.clone();
            if any_equal(
                ptr.si
                    .primitive
                    .clone()
                    .unwrap()
                    .get_material()
                    .unwrap()
                    .as_any(),
                self.material.as_any(),
            ) {
                let next = Arc::new(InteractionChain {
                    si: Arc::new(SurfaceInteraction::default()),
                    next: None,
                });
                Arc::get_mut(&mut ptr).unwrap().next.replace(next.clone());
                ptr = next;
                n_found += 1;
            }
        }

        if n_found == 0 {
            return Spectrum::new(0.0);
        }

        let mut selected = clamp((u1 * n_found as Float) as isize, 0, n_found - 1);
        while {
            let ok = selected > 0;
            selected -= 1;
            ok
        } {
            chain = chain.next.clone().unwrap();
        }
        unsafe { std::ptr::copy(Arc::as_ptr(&chain.si), si as *mut SurfaceInteraction, 1) };
        *pdf = self.pdf_sp(si) / n_found as Float;
        self.sp(si)
    }

    pub fn pdf_sp(&self, pi: &SurfaceInteraction) -> Float {
        let d = self.po.p - pi.p;
        let d_local = Vector3f::new(self.ss.dot(&d), self.ts.dot(&d), self.ns.dot(&d));
        let n_local = Normal3f::new(self.ss.dot(&pi.n), self.ts.dot(&pi.n), self.ns.dot(&pi.n));
        let r_proj = [
            (d_local.y * d_local.y + d_local.z * d_local.z).sqrt(),
            (d_local.z * d_local.z + d_local.x * d_local.x).sqrt(),
            (d_local.x * d_local.x + d_local.y * d_local.y).sqrt(),
        ];
        let mut pdf = 0.0;
        let axis_prob = [0.25, 0.25, 0.5];
        let ch_prob = 1.0 / Spectrum::n_samples() as Float;
        for axis in 0..3 {
            for ch in 0..Spectrum::n_samples() {
                pdf +=
                    self.pdf_sr(ch, r_proj[axis]) * n_local[axis].abs() * ch_prob * axis_prob[axis];
            }
        }
        pdf
    }
}

impl BSSRDF for BaseSeparableBSSRDF {
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
        let ssp = self.sample_sp(scene, u1, u2, si, pdf);
        if !ssp.is_black() {
            let mut bsdf = BSDF::new(si, 1.0);
            bsdf.add(Arc::new(Box::new(SeparableBSSRDFAdapter::new(Arc::new(
                Box::new(self.clone()),
            )))));
            si.bsdf.replace(Arc::new(bsdf));
            si.wo = si.shading.n;
        }
        ssp
    }

    fn eta(&self) -> f32 {
        self.eta
    }
}

impl SeparableBSSRDF for BaseSeparableBSSRDF {
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
pub struct TabulateBSSRDF {
    #[deref]
    #[deref_mut]
    base: BaseSeparableBSSRDF,
    table: Arc<BSSRDFTable>,
    sigma_t: Spectrum,
    rho: Spectrum,
}

impl BSSRDF for TabulateBSSRDF {
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

impl SeparableBSSRDF for TabulateBSSRDF {
    fn sr(&self, r: f32) -> Spectrum {
        let mut ssr = Spectrum::new(0.0);
        for ch in 0..Spectrum::n_samples() {
            let r_optical = r * self.sigma_t[ch];
            let mut rho_offset = 0;
            let mut radius_offset = 0;
            let mut rho_weights = [0.0; 4];
            let mut radius_weights = [0.0; 4];
            if !catmull_rom_weights(
                self.table.n_rho_samples,
                self.table.rho_samples.as_slice(),
                self.rho[ch],
                &mut rho_offset,
                &mut rho_weights,
            ) || !catmull_rom_weights(
                self.table.n_radius_samples,
                self.table.radius_samples.as_slice(),
                r_optical,
                &mut radius_offset,
                &mut radius_weights,
            ) {
                continue;
            }
            let mut sr = 0.0;
            for i in 0..4 {
                for j in 0..4 {
                    let weight = rho_weights[i] * radius_weights[j];
                    if weight != 0.0 {
                        sr += weight * self.table.eval_profile(rho_offset + i, radius_offset + j);
                    }
                }
            }

            if r_optical != 0.0 {
                sr /= 2.0 * PI * r_optical;
            }
            ssr[ch] = sr;
        }
        ssr *= self.sigma_t * self.sigma_t;
        ssr.clamp(0.0, Float::INFINITY)
    }

    fn sampler_sr(&self, ch: usize, u: f32) -> f32 {
        if self.sigma_t[ch] == 0.0 {
            return -1.0;
        }
        sample_catmull_rom_2d(
            self.table.n_rho_samples,
            self.table.n_radius_samples,
            self.table.rho_samples.as_slice(),
            self.table.radius_samples.as_slice(),
            self.table.profile.as_slice(),
            self.table.profile_cdf.as_slice(),
            self.rho[ch],
            u,
            None,
            None,
        ) / self.sigma_t[ch]
    }

    fn pdf_sr(&self, ch: usize, r: f32) -> f32 {
        let r_optical = r * self.sigma_t[ch];
        let mut rho_offset = 0;
        let mut radius_offset = 0;
        let mut rho_weights = [0.0; 4];
        let mut radius_weights = [0.0; 4];
        if !catmull_rom_weights(
            self.table.n_rho_samples,
            self.table.rho_samples.as_slice(),
            self.rho[ch],
            &mut rho_offset,
            &mut rho_weights,
        ) || !catmull_rom_weights(
            self.table.n_radius_samples,
            self.table.radius_samples.as_slice(),
            r_optical,
            &mut radius_offset,
            &mut radius_weights,
        ) {
            return 0.0;
        }

        let mut sr = 0.0;
        let mut rho_eff = 0.0;
        for i in 0..4 {
            if rho_weights[i] == 0.0 {
                continue;
            }
            rho_eff += self.table.rho_eff[rho_offset + i] * rho_weights[i];
            for j in 0..4 {
                if radius_weights[j] == 0.0 {
                    continue;
                }
                sr += self.table.eval_profile(rho_offset + i, radius_offset + j)
                    * rho_weights[i]
                    * radius_weights[j];
            }
        }
        if r_optical != 0.0 {
            sr /= 2.0 * PI * r_optical;
        }

        (sr * self.sigma_t[ch] * self.sigma_t[ch] / rho_eff).max(0.0)
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
        Self {
            n_rho_samples,
            n_radius_samples,
            rho_samples: vec![0.0; n_rho_samples],
            radius_samples: vec![0.0; n_radius_samples],
            profile: vec![0.0; n_radius_samples * n_rho_samples],
            rho_eff: vec![0.0; n_rho_samples],
            profile_cdf: vec![0.0; n_rho_samples * n_radius_samples],
        }
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

impl SeparableBSSRDFAdapter {
    pub fn new(bssrdf: SeparableBSSRDFDt) -> Self {
        Self {
            base: BaseBxDF::new(BxDFType::BSDF_REFLECTION | BxDFType::BSDF_DIFFUSE),
            bssrdf,
        }
    }
}

pub type SeparableBSSRDFDt = Arc<Box<dyn SeparableBSSRDF + Sync + Send>>;

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
    // Compute material parameters and minimum $t$ below the critical angle
    let sigma_t = sigma_a + sigma_s;
    let rho = sigma_s / sigma_t;
    let t_crit = r * (eta * eta - 1.0).sqrt();
    let mut ess = 0.0;
    const N_SAMPLES: usize = 100;
    for i in 0..N_SAMPLES {
        // Evaluate single scattering integrand and add to _Ess_
        let ti = t_crit - (1.0 - (i as Float + 0.5) / N_SAMPLES as Float).ln() / sigma_t;

        // Determine length $d$ of connecting segment and $\cos\theta_\roman{o}$
        let d = (r * r + ti * ti).sqrt();
        let cos_theta_o = ti / d;

        // Add contribution of single scattering at depth $t$
        ess += rho * (-sigma_t * (d + t_crit)).exp() / (d * d)
            * phase_hg(cos_theta_o, g)
            * (1.0 - fr_dielectric(-cos_theta_o, 1.0, eta))
            * cos_theta_o.abs();
    }
    ess / N_SAMPLES as Float
}

pub fn beam_diffusion_ms(sigma_s: Float, sigma_a: Float, g: Float, eta: Float, r: Float) -> Float {
    const N_SAMPLES: usize = 100;
    let mut ed = 0.0;
    // Precompute information for dipole integrand

    // Compute reduced scattering coefficients $\sigmaps, \sigmapt$ and albedo
    // $\rhop$
    let sigmap_s = sigma_s * (1.0 - g);
    let sigmap_t = sigma_a + sigmap_s;
    let rhop = sigmap_s / sigmap_t;

    // Compute non-classical diffusion coefficient $D_\roman{G}$ using
    // Equation (15.24)
    let d_g = (2.0 * sigma_a + sigmap_s) / (3.0 * sigmap_t * sigmap_t);

    // Compute effective transport coefficient $\sigmatr$ based on $D_\roman{G}$
    let sigma_tr = (sigma_a / d_g).sqrt();

    // Determine linear extrapolation distance $\depthextrapolation$ using
    // Equation (15.28)
    let fm1 = fresnel_moment1(eta);
    let fm2 = fresnel_moment2(eta);
    let ze = -2.0 * d_g * (1.0 + 3.0 * fm2) / (1.0 - 2.0 * fm1);

    // Determine exitance scale factors using Equations (15.31) and (15.32)
    let c_phi = 0.25 * (1.0 - 2.0 * fm1);
    let ce = 0.5 * (1.0 - 3.0 * fm2);
    for i in 0..N_SAMPLES {
        // Sample real point source depth $\depthreal$
        let zr = -(1.0 - (i as Float + 0.5) / N_SAMPLES as Float).ln() / sigmap_t;

        // Evaluate dipole integrand $E_{\roman{d}}$ at $\depthreal$ and add to
        // _Ed_
        let zv = -zr + 2.0 * ze;
        let dr = (r * r + zr * zr).sqrt();
        let dv = (r * r + zv * zv).sqrt();

        // Compute dipole fluence rate $\dipole(r)$ using Equation (15.27)
        let phi_d = INV_4_PI / d_g * ((-sigma_tr * dr).exp() / dr - (-sigma_tr * dv).exp() / dv);

        // Compute dipole vector irradiance $-\N{}\cdot\dipoleE(r)$ using
        // Equation (15.27)
        let edn = INV_4_PI
            * (zr * (1.0 + sigma_tr * dr) * (-sigma_tr * dr).exp() / (dr * dr * dr)
                - zv * (1.0 + sigma_tr * dv) * (-sigma_tr * dv).exp() / (dv * dv * dv));

        // Add contribution from dipole for depth $\depthreal$ to _Ed_
        let e = phi_d * c_phi + edn * ce;
        let kappa = 1.0 - (-2.0 * sigmap_t * (dr + zr)).exp();
        ed += kappa * rhop * rhop * e;
    }
    ed / N_SAMPLES as Float
}

pub fn compute_beam_diffusion_bssrdf(g: Float, eta: Float, t: &mut BSSRDFTable) {
    // Choose radius values of the diffusion profile discretization
    t.radius_samples[0] = 0.0;
    t.radius_samples[1] = 2.5e-3;
    for i in 2..t.n_radius_samples {
        t.radius_samples[i] = t.radius_samples[i - 1] * 1.2;
    }

    // Choose albedo values of the diffusion profile discretization
    for i in 0..t.n_rho_samples {
        t.rho_samples[i] = (1.0 - (-8.0 * i as Float / (t.n_rho_samples as Float - 1.0)).exp())
            / (1.0 - (-8.0 as Float).exp());
    }

    //parallel_for!(
    //   |i: usize| {
    for i in 0..t.n_rho_samples {
        // Compute the diffusion profile for the _i_th albedo sample

        // Compute scattering profile for chosen albedo $\rho$
        for j in 0..t.n_radius_samples {
            let rho = t.rho_samples[i];
            let r = t.radius_samples[j];
            t.profile[i * t.n_radius_samples + j] = 2.0
                * PI
                * r
                * (beam_diffusion_ss(rho, 1.0 - rho, g, eta, r)
                    + beam_diffusion_ms(rho, 1.0 - rho, g, eta, r));
        }

        // Compute effective albedo $\rho_{\roman{eff}}$ and CDF for importance
        // sampling
        t.rho_eff[i] = integrate_catmull_rom(
            t.n_radius_samples,
            t.radius_samples.as_slice(),
            &t.profile[i * t.n_radius_samples..],
            &mut t.profile_cdf[i * t.n_radius_samples..],
        );
    }
    // t.n_rho_samples,
    //);
}

pub fn subsurface_from_diffuse(
    t: &BSSRDFTable,
    rho_eff: &Spectrum,
    mfp: &Spectrum,
    sigma_a: &mut Spectrum,
    sigma_s: &mut Spectrum,
) {
    for c in 0..Spectrum::n_samples() {
        let rho = invert_catmull_rom(
            t.n_rho_samples,
            t.rho_samples.as_slice(),
            t.rho_eff.as_slice(),
            rho_eff[c],
        );
        sigma_s[c] = rho / mfp[c];
        sigma_a[c] = (1.0 - rho) / mfp[c];
    }
}
