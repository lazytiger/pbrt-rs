use crate::core::{
    geometry::Vector3f,
    interaction::SurfaceInteraction,
    pbrt::Float,
    texture::{turbulence, Texture, TextureMapping3DDt},
};
use derive_more::{Deref, DerefMut};
use std::{any::Any, fmt::Debug};

#[derive(Debug)]
pub struct WrinkleTexture {
    mapping: TextureMapping3DDt,
    octaves: Float,
    omega: Float,
}

impl WrinkleTexture {
    pub fn new(mapping: TextureMapping3DDt, octaves: Float, omega: Float) -> Self {
        Self {
            mapping,
            octaves,
            omega,
        }
    }
}

impl<T> Texture<T> for WrinkleTexture
where
    T: Debug,
    T: 'static,
    T: From<Float>,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, si: &SurfaceInteraction) -> T {
        let mut dpdx = Vector3f::default();
        let mut dpdy = Vector3f::default();
        let p = self.mapping.map(si, &mut dpdx, &mut dpdy);
        turbulence(&p, &dpdx, &dpdy, self.omega, self.octaves).into()
    }
}
