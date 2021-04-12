use crate::{
    core::{
        filter::{BaseFilter, Filter},
        geometry::{Point2f, Vector2f},
        pbrt::Float,
    },
    impl_base_filter,
};
use std::f32::consts::PI;

pub struct LanczosSincFilter {
    base: BaseFilter,
    tau: Float,
}

impl LanczosSincFilter {
    pub fn new(radius: Vector2f, tau: Float) -> Self {
        Self {
            base: BaseFilter::new(radius),
            tau,
        }
    }

    pub fn sinc(&self, x: Float) -> Float {
        let x = x.abs();
        if x < 1e-5 {
            1.0
        } else {
            (PI * x).sin() / (PI * x)
        }
    }

    pub fn windowed_sinc(&self, x: Float, radius: Float) -> Float {
        let x = x.abs();
        if x > radius {
            0.0
        } else {
            let lanczos = self.sinc(x / self.tau);
            self.sinc(x) * lanczos
        }
    }
}

impl Filter for LanczosSincFilter {
    impl_base_filter!();

    fn evaluate(&self, p: &Point2f) -> f32 {
        self.windowed_sinc(p.x, self.base.radius.x) * self.windowed_sinc(p.y, self.base.radius.y)
    }
}
