use crate::core::{
    interaction::SurfaceInteraction,
    reflection::ScaledBxDF,
    texture::{Texture, TextureDt},
};
use derive_more::{Deref, DerefMut};
use std::{any::Any, ops::Mul};

pub struct ScaleTexture<T1, T2> {
    tex1: TextureDt<T1>,
    tex2: TextureDt<T2>,
}

impl<T1, T2> ScaleTexture<T1, T2> {
    pub fn new(tex1: TextureDt<T1>, tex2: TextureDt<T2>) -> Self {
        Self { tex1, tex2 }
    }
}

impl<T1, T2> Texture<T2> for ScaleTexture<T1, T2>
where
    T1: 'static,
    T2: 'static,
    T2: Mul<T1, Output = T2>,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, si: &SurfaceInteraction) -> T2 {
        self.tex2.evaluate(si) * self.tex1.evaluate(si)
    }
}
