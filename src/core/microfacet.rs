use crate::core::{
    geometry::{Point2f, Vector3f},
    pbrt::Float,
    reflection::abs_cos_theta,
};
use std::sync::Arc;

pub trait MicrofacetDistribution {
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
