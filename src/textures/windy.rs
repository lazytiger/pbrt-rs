use crate::core::{
    geometry::Vector3f,
    interaction::SurfaceInteraction,
    pbrt::Float,
    texture::{fbm, Texture, TextureMapping3DDt},
};
use derive_more::{Deref, DerefMut};
use std::any::Any;

pub struct WindyTexture {
    mapping: TextureMapping3DDt,
}

impl WindyTexture {
    pub fn new(mapping: TextureMapping3DDt) -> Self {
        Self { mapping }
    }
}

impl<T> Texture<T> for WindyTexture
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
        let wind_strength = fbm(&(p * 0.1), &(dpdx * 0.1), &(dpdy * 0.1), 0.5, 3.0);
        let wave_height = fbm(&p, &dpdx, &dpdy, 0.5, 6.0);
        (wind_strength.abs() * wave_height).into()
    }
}
