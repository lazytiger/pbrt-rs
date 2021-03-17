use crate::core::geometry::{Point2f, Vector3f};
use crate::Float;

pub trait PhaseFunction {
    fn p(&self, wo: &Vector3f, wi: &Vector3f) -> Float;
    fn sample_p(&self, wo: &Vector3f, wi: &mut Vector3f, u: &Point2f) -> Float;
    fn to_string(&self) -> String;
}
pub struct Medium {}

pub struct MediumInterface {}
