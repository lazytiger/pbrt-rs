use crate::core::{
    interaction::SurfaceInteraction,
    pbrt::Float,
    texture::{Texture, TextureDt},
};
use derive_more::{Deref, DerefMut};
use std::{
    any::Any,
    ops::{Add, Mul},
    process::Output,
};

pub struct MixTexture<T> {
    tex1: TextureDt<T>,
    tex2: TextureDt<T>,
    amount: TextureDt<Float>,
}

impl<T> MixTexture<T> {
    pub fn new(tex1: TextureDt<T>, tex2: TextureDt<T>, amount: TextureDt<Float>) -> Self {
        Self { tex1, tex2, amount }
    }
}

impl<T> Texture<T> for MixTexture<T>
where
    T: 'static,
    T: Mul<Float, Output = T>,
    T: Add<Output = T>,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, si: &SurfaceInteraction) -> T {
        let t1 = self.tex1.evaluate(si);
        let t2 = self.tex2.evaluate(si);
        let amt = self.amount.evaluate(si);
        t1 * (1.0 - amt) + t2 * amt
    }
}
