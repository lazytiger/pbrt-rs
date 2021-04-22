use crate::core::{
    geometry::Vector3f,
    interaction::SurfaceInteraction,
    pbrt::Float,
    spectrum::Spectrum,
    texture::{fbm, Texture, TextureMapping3DDt},
};
use derive_more::{Deref, DerefMut};
use std::any::Any;

pub struct FBmTexture {
    mapping: TextureMapping3DDt,
    omega: Float,
    octaves: Float,
}

impl FBmTexture {
    pub fn new(mapping: TextureMapping3DDt, octaves: Float, omega: Float) -> Self {
        Self {
            mapping,
            octaves,
            omega,
        }
    }
}

impl<T> Texture<T> for FBmTexture
where
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
        fbm(&p, &dpdx, &dpdy, self.omega, self.octaves).into()
    }
}
