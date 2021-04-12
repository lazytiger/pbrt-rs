use crate::{
    core::{
        filter::{BaseFilter, Filter},
        geometry::{Point2f, Vector2f},
        pbrt::Float,
    },
    impl_base_filter,
};

pub struct GaussianFilter {
    base: BaseFilter,
    alpha: Float,
    exp_x: Float,
    exp_y: Float,
}

impl GaussianFilter {
    pub fn new(radius: Vector2f, alpha: Float) -> Self {
        let exp_x = (-alpha * radius.x * radius.x).exp();
        let exp_y = (-alpha * radius.y * radius.y).exp();
        GaussianFilter {
            base: BaseFilter::new(radius),
            alpha,
            exp_x,
            exp_y,
        }
    }

    fn gaussian(&self, d: Float, exp_v: Float) -> Float {
        ((-self.alpha * d * d).exp() - exp_v).max(0.0)
    }
}

impl Filter for GaussianFilter {
    impl_base_filter!();

    fn evaluate(&self, p: &Point2f) -> Float {
        self.gaussian(p.x, self.exp_x) * self.gaussian(p.y, self.exp_y)
    }
}
