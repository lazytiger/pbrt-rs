use crate::core::{
    geometry::Vector2f,
    interaction::SurfaceInteraction,
    pbrt::Float,
    spectrum::Spectrum,
    texture::{Texture, TextureMapping2DDt, TextureMapping3D},
};
use derive_more::{Deref, DerefMut};
use std::{
    any::Any,
    marker::PhantomData,
    ops::{Add, Mul},
};

pub struct BilerpTexture<T> {
    mapping: TextureMapping2DDt,
    v00: T,
    v01: T,
    v10: T,
    v11: T,
}

impl<T> BilerpTexture<T> {
    pub fn new(mapping: TextureMapping2DDt, v00: T, v01: T, v02: T, v10: T, v11: T) -> Self {
        Self {
            mapping,
            v00,
            v01,
            v10,
            v11,
        }
    }
}

impl<T> Texture<T> for BilerpTexture<T>
where
    T: 'static,
    T: Add<Output = T>,
    for<'a> &'a T: Mul<Float, Output = T>,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, si: &SurfaceInteraction) -> T {
        let mut dstdx = Vector2f::default();
        let mut dstdy = Vector2f::default();
        let st = self.mapping.map(si, &mut dstdx, &mut dstdy);
        &self.v00 * ((1.0 - st[0]) * (1.0 - st[1]))
            + &self.v01 * ((1.0 - st[0]) * st[1])
            + &self.v10 * (st[0] * (1.0 - st[1]))
            + &self.v11 * (st[0] * st[1])
    }
}

pub type BilerpTexturef = BilerpTexture<Float>;
pub type BilerpTextureS = BilerpTexture<Spectrum>;
