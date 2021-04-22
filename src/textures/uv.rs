use crate::core::{
    geometry::Vector2f,
    interaction::SurfaceInteraction,
    spectrum::{Spectrum, SpectrumType},
    texture::{Texture, TextureMapping2DDt},
};
use derive_more::{Deref, DerefMut};
use std::any::Any;

pub struct UVTexture {
    mapping: TextureMapping2DDt,
}

impl UVTexture {
    pub fn new(mapping: TextureMapping2DDt) -> Self {
        Self { mapping }
    }
}

impl Texture<Spectrum> for UVTexture {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, si: &SurfaceInteraction) -> Spectrum {
        let mut dstdx = Vector2f::default();
        let mut dstdy = Vector2f::default();
        let st = self.mapping.map(si, &mut dstdx, &mut dstdy);
        let rgb = [st[0] - st[0].floor(), st[1] - st[1].floor(), 0.0];
        Spectrum::from_rgb(&rgb, SpectrumType::Reflectance)
    }
}
