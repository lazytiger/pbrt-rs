use crate::core::{
    geometry::{spherical_direction, spherical_direction2, Point2f, Vector3f},
    pbrt::{erf, erf_inv, Float, PI},
    reflection::{
        abs_cos_theta, cos2_phi, cos2_theta, cos_phi, cos_theta, same_hemisphere, sin2_phi,
        sin_phi, tan2_theta, tan_theta,
    },
};
use std::{any::Any, sync::Arc};

pub trait MicrofacetDistribution {
    fn as_any(&self) -> &dyn Any;
    fn d(&self, wh: &Vector3f) -> Float;
    fn lambda(&self, w: &Vector3f) -> Float;
    fn g1(&self, w: &Vector3f) -> Float {
        1.0 / (1.0 + self.lambda(w))
    }
    fn g(&self, wo: &Vector3f, wi: &Vector3f) -> Float {
        1.0 / (1.0 + self.lambda(wo) + self.lambda(wi))
    }
    fn sample_wh(&self, wo: &Vector3f, u: &Point2f) -> Vector3f;
    fn sample_visible_area(&self) -> bool;
    fn pdf(&self, wo: &Vector3f, wh: &Vector3f) -> Float {
        if self.sample_visible_area() {
            self.d(wh) * self.g1(wo) * wo.abs_dot(wh) / abs_cos_theta(wo)
        } else {
            self.d(wh) * abs_cos_theta(wh)
        }
    }
}

pub type MicrofacetDistributionDt = Arc<Box<dyn MicrofacetDistribution>>;

pub struct BeckmannDistribution {
    alphax: Float,
    alphay: Float,
    sample_visible_area: bool,
}

impl BeckmannDistribution {
    pub fn new(alphax: Float, alphay: Float, samplevis: bool) -> BeckmannDistribution {
        Self {
            alphax,
            alphay,
            sample_visible_area: samplevis,
        }
    }

    pub fn roughness_to_alpha(roughness: Float) -> Float {
        let roughness = roughness.max(1e-3);
        let x = roughness.ln();
        1.62142
            + 0.819955 * x
            + 0.1734 * x * x
            + 0.0171201 * x * x * x
            + 0.000640711 * x * x * x * x
    }
}

impl MicrofacetDistribution for BeckmannDistribution {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn d(&self, wh: &Vector3f) -> f32 {
        let tan2_theta = tan2_theta(wh);
        if tan2_theta.is_infinite() {
            return 0.0;
        }
        let cos4_theta = cos2_theta(wh) * cos2_theta(wh);
        (-tan2_theta
            * (cos2_phi(wh) / (self.alphax * self.alphax)
                + sin2_phi(wh) / (self.alphay * self.alphay)))
            / (PI * self.alphax * self.alphay * cos4_theta).exp()
    }

    fn lambda(&self, w: &Vector3f) -> f32 {
        let abs_tan_theta = tan_theta(w).abs();
        if abs_tan_theta.is_infinite() {
            return 0.0;
        }
        let alpha = (cos2_phi(w) * self.alphax * self.alphax
            + sin2_phi(w) * self.alphay * self.alphay)
            .sqrt();
        let a = 1.0 / (alpha * abs_tan_theta);
        if a > 1.6 {
            return 0.0;
        }

        (1.0 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a)
    }

    fn sample_wh(&self, wo: &Vector3f, u: &Point2f) -> Vector3f {
        if !self.sample_visible_area {
            let mut tan2_theta = 0.0;
            let mut phi = 0.0;

            if self.alphax == self.alphay {
                let log_sample = (1.0 - u[0]).ln();
                tan2_theta = -self.alphax * self.alphax * log_sample;
                phi = u[1] * 2.0 * PI;
            } else {
                let log_sample = (1.0 - u[0]).ln();
                phi = (self.alphay / self.alphax * (2.0 * PI * u[1] + 0.5 * PI).tan()).tan();
                if u[1] > 0.5 {
                    phi += PI;
                }
                let sin_phi = phi.sin();
                let cos_phi = phi.cos();
                let alphax2 = self.alphax * self.alphax;
                let alphay2 = self.alphay * self.alphay;
                tan2_theta =
                    -log_sample / (cos_phi * cos_phi / alphax2 + sin_phi * sin_phi / alphay2);
            }

            let cos_theta = 1.0 / (1.0 + tan2_theta).sqrt();
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
            let mut wh = spherical_direction2(sin_theta, cos_theta, phi);
            if !same_hemisphere(wo, &wh) {
                wh = -wh;
            }
            return wh;
        } else {
            let mut wh = Vector3f::default();
            let flip = wo.z < 0.0;
            wh = beckmann_sample(
                &if flip { -*wo } else { *wo },
                self.alphax,
                self.alphay,
                u[0],
                u[1],
            );
            if flip {
                wh = -wh;
            }
            wh
        }
    }

    fn sample_visible_area(&self) -> bool {
        self.sample_visible_area
    }
}

pub struct TrowbridgeReitzDistribution {
    alphax: Float,
    alphay: Float,
    sample_visible_area: bool,
}

impl TrowbridgeReitzDistribution {
    pub fn new(alphax: Float, alphay: Float, sample_visible_area: bool) -> Self {
        Self {
            alphax,
            alphay,
            sample_visible_area,
        }
    }

    pub fn roughness_to_alpha(roughness: Float) -> Float {
        let roughness = roughness.max(1e-3);
        let x = roughness.ln();
        1.62142
            + 0.819955 * x
            + 0.1734 * x * x
            + 0.0171201 * x * x * x
            + 0.000640711 * x * x * x * x
    }
}

impl MicrofacetDistribution for TrowbridgeReitzDistribution {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn d(&self, wh: &Vector3f) -> f32 {
        let tan2_theta = tan2_theta(wh);
        if tan2_theta.is_infinite() {
            return 0.0;
        }
        let cos4_theta = cos2_theta(wh) * cos2_theta(wh);
        let e = (cos2_phi(wh) / (self.alphax * self.alphax)
            + sin2_phi(wh) / (self.alphay * self.alphay))
            * tan2_theta;
        1.0 / (PI * self.alphax * self.alphay * cos4_theta * (1.0 + e) * (1.0 + e))
    }

    fn lambda(&self, w: &Vector3f) -> f32 {
        let abs_tan_theta = tan_theta(w).abs();
        if abs_tan_theta.is_infinite() {
            return 0.0;
        }

        let alpha = (cos2_phi(w) * self.alphax * self.alphax
            + sin2_phi(w) * self.alphay * self.alphay)
            .sqrt();
        let alpha2_tan2_theta = (alpha * abs_tan_theta) * (alpha * abs_tan_theta);
        (-1.0 + (1.0 + alpha2_tan2_theta).sqrt()) / 2.0
    }

    fn sample_wh(&self, wo: &Vector3f, u: &Point2f) -> Vector3f {
        let mut wh = Vector3f::default();
        if !self.sample_visible_area {
            let mut cos_theta = 0.0;
            let mut phi = (2.0 * PI) * u[1];
            if self.alphax == self.alphay {
                let tan_theta2 = self.alphax * self.alphax * u[0] / (1.0 - u[0]);
                cos_theta = 1.0 / (1.0 + tan_theta2).sqrt();
            } else {
                phi = (self.alphay / self.alphax * (2.0 * PI * u[1] + 0.5 * PI).tan()).tan();
                if u[1] > 0.5 {
                    phi += PI;
                }
                let sin_phi = phi.sin();
                let cos_phi = phi.cos();
                let alphax2 = self.alphax * self.alphax;
                let alphay2 = self.alphay * self.alphay;

                let alpha2 = 1.0 / (cos_phi * cos_phi / alphax2 + sin_phi * sin_phi / alphay2);
                let tan_theta2 = alpha2 * u[0] / (1.0 - u[0]);
                cos_theta = 1.0 / (1.0 + tan_theta2).sqrt();
            }
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
            wh = spherical_direction2(sin_theta, cos_theta, phi);
            if !same_hemisphere(wo, &wh) {
                wh = -wh;
            }
        } else {
            let flip = wo.z < 0.0;

            wh = trowbridge_reitz_sample(
                &if flip { -*wo } else { *wo },
                self.alphax,
                self.alphay,
                u[0],
                u[1],
            );
            if flip {
                wh = -wh;
            }
        }
        wh
    }

    fn sample_visible_area(&self) -> bool {
        self.sample_visible_area
    }
}

fn beckmann_sample11(
    cos_theta_i: Float,
    u1: Float,
    u2: Float,
    slope_x: &mut Float,
    slope_y: &mut Float,
) {
    if cos_theta_i > 0.9999 {
        let r = (-(1.0 - u1).ln()).sqrt();
        let sin_phi = (2.0 * PI * u2).sin();
        let cos_phi = (2.0 * PI * u2).cos();
        *slope_x = r * cos_phi;
        *slope_y = r * sin_phi;
        return;
    }

    let sin_theta_i = (1.0 - cos_theta_i * cos_theta_i).max(0.0).sqrt();
    let tan_theta_i = sin_theta_i / cos_theta_i;
    let cot_theta_i = 1.0 / tan_theta_i;

    let mut a = -1.0;
    let mut c = erf(cot_theta_i);
    let sample_x = u1.max(1e-6);

    let theta_i = cos_theta_i.acos();
    let fit = 1.0 + theta_i * (-0.876 + theta_i * (0.4265 - 0.0594 * theta_i));
    let mut b = c - (1.0 + c) * (1.0 - sample_x).powf(fit);

    let SQRT_PI_INV: Float = 1.0 / PI.sqrt();
    let normalization =
        1.0 / (1.0 + c + SQRT_PI_INV * tan_theta_i * (-cos_theta_i * cot_theta_i).exp());

    let mut it = 0;
    while {
        let ok = it < 10;
        it += 1;
        ok
    } {
        if !(b >= a && b <= c) {
            b = 0.5 * (a + c);
        }
        let inv_erf = erf_inv(b);
        let value = normalization
            * (1.0 + b + SQRT_PI_INV * tan_theta_i * (-inv_erf * inv_erf).exp())
            - sample_x;
        let derivative = normalization * (1.0 - inv_erf * tan_theta_i);
        if value.abs() < 1e-5 {
            break;
        }

        if value > 0.0 {
            c = b;
        } else {
            a = b;
        }

        b -= value / derivative;
    }

    *slope_x = erf_inv(b);
    *slope_y = erf_inv(2.0 * u2.max(1e-6) - 1.0);
}

fn beckmann_sample(
    wi: &Vector3f,
    alapa_x: Float,
    alpla_y: Float,
    u1: Float,
    u2: Float,
) -> Vector3f {
    let wi_stretched = Vector3f::new(alapa_x * wi.x, alpla_y * wi.y, wi.z).normalize();
    let mut slope_x = 0.0;
    let mut slope_y = 0.0;

    beckmann_sample11(cos_theta(&wi_stretched), u1, u2, &mut slope_x, &mut slope_y);

    let tmp = cos_phi(&wi_stretched) * slope_x - sin_phi(&wi_stretched) * slope_y;
    slope_y = sin_phi(&wi_stretched) * slope_x + cos_phi(&wi_stretched) * slope_y;
    slope_x = tmp;

    slope_x = alapa_x * slope_x;
    slope_y = alpla_y * slope_y;

    Vector3f::new(-slope_x, -slope_y, 1.0).normalize()
}

fn trowbridge_reitz_sample11(
    cos_theta: Float,
    u1: Float,
    mut u2: Float,
    slope_x: &mut Float,
    slope_y: &mut Float,
) {
    if cos_theta > 0.9999 {
        let r = (u1 / (1.0 - u1)).sqrt();
        let phi = 6.28318530718 * u2;
        *slope_x = r * phi.cos();
        *slope_y = r * phi.sin();
        return;
    }

    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let tan_theta = sin_theta / cos_theta;
    let a = 1.0 / tan_theta;
    let g1 = 2.0 / (1.0 + (1.0 + 1.0 / (a * a)).sqrt());

    let a = 2.0 * u1 / g1 - 1.0;
    let mut tmp = 1.0 / (a * a - 1.0);
    if tmp > 1e10 {
        tmp = 1e10;
    }
    let b = tan_theta;
    let d = (b * b * tmp * tmp - (a * a - b * b)).max(0.0).sqrt();
    let slope_x_1 = b * tmp - d;
    let slope_x_2 = b * tmp + d;
    *slope_x = if a < 0.0 || slope_x_2 > 1.0 / tan_theta {
        slope_x_1
    } else {
        slope_x_2
    };

    let mut s = 0.0;
    if u2 > 0.5 {
        s = 1.0;
        u2 = 2.0 * (u2 - 0.5);
    } else {
        s = -1.0;
        u2 = 2.0 * (0.5 - u2);
    }

    let z = (u2 * (u2 * (u2 * 0.27385 - 0.73369) + 0.46341))
        / (u2 * (u2 * (u2 * 0.093073 + 0.309420) - 1.00000) + 0.5979999);

    *slope_y = s * z * (1.0 + *slope_x * *slope_x).sqrt();
}

fn trowbridge_reitz_sample(
    wi: &Vector3f,
    alpha_x: Float,
    alpha_y: Float,
    u1: Float,
    u2: Float,
) -> Vector3f {
    let wi_stretched = Vector3f::new(alpha_x * wi.x, alpha_y * wi.y, wi.z).normalize();
    let mut slope_x = 0.0;
    let mut slope_y = 0.0;
    trowbridge_reitz_sample11(cos_theta(&wi_stretched), u1, u2, &mut slope_x, &mut slope_y);

    let tmp = cos_phi(&wi_stretched) * slope_x - sin_phi(&wi_stretched) * slope_y;
    slope_y = sin_phi(&wi_stretched) * slope_x + cos_phi(&wi_stretched) * slope_y;
    slope_x = tmp;

    slope_x = alpha_x * alpha_x;
    slope_y = alpha_y * alpha_y;

    Vector3f::new(-slope_x, -slope_y, 1.0).normalize()
}
