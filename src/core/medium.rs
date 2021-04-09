use crate::core::geometry::{Point2f, Vector3f};
use crate::core::pbrt::{any_equal, Float};
use std::any::Any;
use std::sync::Arc;

pub trait PhaseFunction {
    fn as_any(&self) -> &dyn Any;
    fn p(&self, wo: &Vector3f, wi: &Vector3f) -> Float;
    fn sample_p(&self, wo: &Vector3f, wi: &mut Vector3f, u: &Point2f) -> Float;
    fn to_string(&self) -> String;
}
pub trait Medium {
    fn as_any(&self) -> &dyn Any;
}

#[derive(Default, Clone)]
pub struct MediumInterface {
    pub inside: Option<Arc<Box<dyn Medium>>>,
    pub outside: Option<Arc<Box<dyn Medium>>>,
}

impl MediumInterface {
    pub fn new(
        inside: Option<Arc<Box<dyn Medium>>>,
        outside: Option<Arc<Box<dyn Medium>>>,
    ) -> MediumInterface {
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
