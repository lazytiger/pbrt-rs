use crate::core::{
    geometry::{spherical_direction, Point2f, Ray, Vector3f},
    interaction::MediumInteraction,
    pbrt::{any_equal, Float, INV_4_PI, PI},
    reflection::cos2_theta,
    sampler::SamplerDtRw,
    spectrum::{Spectrum, SpectrumType::Reflectance},
};
use std::{
    any::Any,
    fmt::Debug,
    sync::{Arc, Mutex, RwLock},
};

pub trait PhaseFunction {
    fn as_any(&self) -> &dyn Any;
    fn p(&self, wo: &Vector3f, wi: &Vector3f) -> Float;
    fn sample_p(&self, wo: &Vector3f, wi: &mut Vector3f, u: &Point2f) -> Float;
}

pub fn get_medium_scattering_properties(
    name: String,
    sigma_a: &mut Spectrum,
    sigma_s: &mut Spectrum,
) -> bool {
    for mss in SUBSURFACE_PARAMETER_TABLE {
        if name == mss.name {
            *sigma_a = Spectrum::from_rgb(&mss.sigma_a, Reflectance);
            *sigma_s = Spectrum::from_rgb(&mss.sigma_prime_s, Reflectance);
            return true;
        }
    }
    false
}

pub fn phase_hg(cos_theta: Float, g: Float) -> Float {
    let denom = 1.0 + g * g + 2.0 * g * cos_theta;
    INV_4_PI * (1.0 - g * g) / (denom * denom.sqrt())
}

pub trait Medium: Debug {
    fn as_any(&self) -> &dyn Any;
    fn tr(&self, ray: &Ray, sampler: SamplerDtRw) -> Spectrum;
    fn sample(&self, ray: &Ray, sampler: SamplerDtRw, mi: &mut MediumInteraction) -> Spectrum;
}

pub type MediumDt = Arc<Box<dyn Medium + Sync + Send>>;
pub type MediumDtMut = Arc<Mutex<Box<dyn Medium + Sync + Send>>>;
pub type MediumDtRw = Arc<RwLock<Box<dyn Medium + Sync + Send>>>;

#[derive(Default, Clone)]
pub struct HenyeyGreenstein {
    g: Float,
}

impl HenyeyGreenstein {
    pub fn new(g: Float) -> Self {
        Self { g }
    }
}

impl PhaseFunction for HenyeyGreenstein {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn p(&self, wo: &Vector3f, wi: &Vector3f) -> f32 {
        phase_hg(wo.dot(wi), self.g)
    }

    fn sample_p(&self, wo: &Vector3f, wi: &mut Vector3f, u: &Point2f) -> f32 {
        let mut cos_theta = 0.0;
        if self.g.abs() < 1e-3 {
            cos_theta = 1.0 - 2.0 * u[0];
        } else {
            let sqr_term = (1.0 - self.g * self.g) / (1.0 + self.g - 2.0 * self.g * u[0]);
            cos_theta = -(1.0 + self.g * self.g - sqr_term * sqr_term) / (2.0 * self.g);
        }
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = 2.0 * PI * u[1];
        let (v1, v2) = wo.coordinate_system();
        *wi = spherical_direction(sin_theta, cos_theta, phi, v1, v2, *wo);
        phase_hg(cos_theta, self.g)
    }
}

#[derive(Default, Clone, Debug)]
pub struct MediumInterface {
    pub inside: Option<MediumDt>,
    pub outside: Option<MediumDt>,
}

impl From<Option<MediumDt>> for MediumInterface {
    fn from(side: Option<MediumDt>) -> Self {
        Self::new(side.clone(), side)
    }
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

struct MeasuredSS {
    name: &'static str,
    sigma_prime_s: [Float; 3],
    sigma_a: [Float; 3],
}
impl MeasuredSS {
    const fn new(name: &'static str, sigma_prime_s: [Float; 3], sigma_a: [Float; 3]) -> Self {
        Self {
            name,
            sigma_a,
            sigma_prime_s,
        }
    }
}

static SUBSURFACE_PARAMETER_TABLE: &[MeasuredSS] = &[
    MeasuredSS::new("Apple", [2.29, 2.39, 1.97], [0.0030, 0.0034, 0.046]),
    MeasuredSS::new("Chicken1", [0.15, 0.21, 0.38], [0.015, 0.077, 0.19]),
    MeasuredSS::new("Chicken2", [0.19, 0.25, 0.32], [0.018, 0.088, 0.20]),
    MeasuredSS::new("Cream", [7.38, 5.47, 3.15], [0.0002, 0.0028, 0.0163]),
    MeasuredSS::new("Ketchup", [0.18, 0.07, 0.03], [0.061, 0.97, 1.45]),
    MeasuredSS::new("Marble", [2.19, 2.62, 3.00], [0.0021, 0.0041, 0.0071]),
    MeasuredSS::new("Potato", [0.68, 0.70, 0.55], [0.0024, 0.0090, 0.12]),
    MeasuredSS::new("Skimmilk", [0.70, 1.22, 1.90], [0.0014, 0.0025, 0.0142]),
    MeasuredSS::new("Skin1", [0.74, 0.88, 1.01], [0.032, 0.17, 0.48]),
    MeasuredSS::new("Skin2", [1.09, 1.59, 1.79], [0.013, 0.070, 0.145]),
    MeasuredSS::new("Spectralon", [11.6, 20.4, 14.9], [0.00, 0.00, 0.00]),
    MeasuredSS::new("Wholemilk", [2.55, 3.21, 3.77], [0.0011, 0.0024, 0.014]),
    // From "Acquiring Scattering Properties of Participating Media by
    // Dilution",
    // Narasimhan, Gupta, Donner, Ramamoorthi, Nayar, Jensen
    // Proc SIGGRAPH 2006
    MeasuredSS::new(
        "Lowfat Milk",
        [0.89187, 1.5136, 2.532],
        [0.002875, 0.00575, 0.0115],
    ),
    MeasuredSS::new(
        "Reduced Milk",
        [2.4858, 3.1669, 4.5214],
        [0.0025556, 0.0051111, 0.012778],
    ),
    MeasuredSS::new(
        "Regular Milk",
        [4.5513, 5.8294, 7.136],
        [0.0015333, 0.0046, 0.019933],
    ),
    MeasuredSS::new(
        "Espresso",
        [0.72378, 0.84557, 1.0247],
        [4.7984, 6.5751, 8.8493],
    ),
    MeasuredSS::new(
        "Mint Mocha Coffee",
        [0.31602, 0.38538, 0.48131],
        [3.772, 5.8228, 7.82],
    ),
    MeasuredSS::new(
        "Lowfat Soy Milk",
        [0.30576, 0.34233, 0.61664],
        [0.0014375, 0.0071875, 0.035937],
    ),
    MeasuredSS::new(
        "Regular Soy Milk",
        [0.59223, 0.73866, 1.4693],
        [0.0019167, 0.0095833, 0.065167],
    ),
    MeasuredSS::new(
        "Lowfat Chocolate Milk",
        [0.64925, 0.83916, 1.1057],
        [0.0115, 0.0368, 0.1564],
    ),
    MeasuredSS::new(
        "Regular Chocolate Milk",
        [1.4585, 2.1289, 2.9527],
        [0.010063, 0.043125, 0.14375],
    ),
    MeasuredSS::new(
        "Coke",
        [8.9053e-05, 8.372e-05, 0.0],
        [0.10014, 0.16503, 0.2468],
    ),
    MeasuredSS::new(
        "Pepsi",
        [6.1697e-05, 4.2564e-05, 0.0],
        [0.091641, 0.14158, 0.20729],
    ),
    MeasuredSS::new(
        "Sprite",
        [6.0306e-06, 6.4139e-06, 6.5504e-06],
        [0.001886, 0.0018308, 0.0020025],
    ),
    MeasuredSS::new(
        "Gatorade",
        [0.0024574, 0.003007, 0.0037325],
        [0.024794, 0.019289, 0.008878],
    ),
    MeasuredSS::new(
        "Chardonnay",
        [1.7982e-05, 1.3758e-05, 1.2023e-05],
        [0.010782, 0.011855, 0.023997],
    ),
    MeasuredSS::new(
        "White Zinfandel",
        [1.7501e-05, 1.9069e-05, 1.288e-05],
        [0.012072, 0.016184, 0.019843],
    ),
    MeasuredSS::new(
        "Merlot",
        [2.1129e-05, 0.0, 0.0],
        [0.11632, 0.25191, 0.29434],
    ),
    MeasuredSS::new(
        "Budweiser Beer",
        [2.4356e-05, 2.4079e-05, 1.0564e-05],
        [0.011492, 0.024911, 0.057786],
    ),
    MeasuredSS::new(
        "Coors Light Beer",
        [5.0922e-05, 4.301e-05, 0.0],
        [0.006164, 0.013984, 0.034983],
    ),
    MeasuredSS::new(
        "Clorox",
        [0.0024035, 0.0031373, 0.003991],
        [0.0033542, 0.014892, 0.026297],
    ),
    MeasuredSS::new(
        "Apple Juice",
        [0.00013612, 0.00015836, 0.000227],
        [0.012957, 0.023741, 0.052184],
    ),
    MeasuredSS::new(
        "Cranberry Juice",
        [0.00010402, 0.00011646, 7.8139e-05],
        [0.039437, 0.094223, 0.12426],
    ),
    MeasuredSS::new(
        "Grape Juice",
        [5.382e-05, 0.0, 0.0],
        [0.10404, 0.23958, 0.29325],
    ),
    MeasuredSS::new(
        "Ruby Grapefruit Juice",
        [0.011002, 0.010927, 0.011036],
        [0.085867, 0.18314, 0.25262],
    ),
    MeasuredSS::new(
        "White Grapefruit Juice",
        [0.22826, 0.23998, 0.32748],
        [0.0138, 0.018831, 0.056781],
    ),
    MeasuredSS::new(
        "Shampoo",
        [0.0007176, 0.0008303, 0.0009016],
        [0.014107, 0.045693, 0.061717],
    ),
    MeasuredSS::new(
        "Strawberry Shampoo",
        [0.00015671, 0.00015947, 1.518e-05],
        [0.01449, 0.05796, 0.075823],
    ),
    MeasuredSS::new(
        "Head & Shoulders Shampoo",
        [0.023805, 0.028804, 0.034306],
        [0.084621, 0.15688, 0.20365],
    ),
    MeasuredSS::new(
        "Lemon Tea Powder",
        [0.040224, 0.045264, 0.051081],
        [2.4288, 4.5757, 7.2127],
    ),
    MeasuredSS::new(
        "Orange Powder",
        [0.00015617, 0.00017482, 0.0001762],
        [0.001449, 0.003441, 0.007863],
    ),
    MeasuredSS::new(
        "Pink Lemonade Powder",
        [0.00012103, 0.00013073, 0.00012528],
        [0.001165, 0.002366, 0.003195],
    ),
    MeasuredSS::new(
        "Cappuccino Powder",
        [1.8436, 2.5851, 2.1662],
        [35.844, 49.547, 61.084],
    ),
    MeasuredSS::new(
        "Salt Powder",
        [0.027333, 0.032451, 0.031979],
        [0.28415, 0.3257, 0.34148],
    ),
    MeasuredSS::new(
        "Sugar Powder",
        [0.00022272, 0.00025513, 0.000271],
        [0.012638, 0.031051, 0.050124],
    ),
    MeasuredSS::new(
        "Suisse Mocha Powder",
        [2.7979, 3.5452, 4.3365],
        [17.502, 27.004, 35.433],
    ),
    MeasuredSS::new(
        "Pacific Ocean Surface Water",
        [0.0001764, 0.00032095, 0.00019617],
        [0.031845, 0.031324, 0.030147],
    ),
];
