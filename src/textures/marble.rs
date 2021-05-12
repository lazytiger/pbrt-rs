use crate::core::{
    geometry::Vector3f,
    interaction::SurfaceInteraction,
    pbrt::Float,
    spectrum::{Spectrum, SpectrumType},
    texture::{fbm, Texture, TextureMapping3DDt},
};
use derive_more::{Deref, DerefMut};
use std::any::Any;

#[derive(Debug)]
pub struct MarbleTexture {
    mapping: TextureMapping3DDt,
    octaves: Float,
    omega: Float,
    scale: Float,
    variation: Float,
}

impl MarbleTexture {
    pub fn new(
        mapping: TextureMapping3DDt,
        octaves: Float,
        omega: Float,
        scale: Float,
        variation: Float,
    ) -> Self {
        Self {
            mapping,
            octaves,
            omega,
            scale,
            variation,
        }
    }
}

impl Texture<Spectrum> for MarbleTexture {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, si: &SurfaceInteraction) -> Spectrum {
        let mut dpdx = Vector3f::default();
        let mut dpdy = Vector3f::default();
        let mut p = self.mapping.map(si, &mut dpdx, &mut dpdy);
        p *= self.scale;
        let marble = p.y
            + self.variation
                * fbm(
                    &p,
                    &(dpdx * self.scale),
                    &(dpdy * self.scale),
                    self.omega,
                    self.octaves,
                );
        let mut t = 0.5 + 0.5 * marble.sin();
        let c = [
            [0.58, 0.58, 0.6],
            [0.58, 0.58, 0.6],
            [0.58, 0.58, 0.6],
            [0.5, 0.5, 0.5],
            [0.6, 0.59, 0.58],
            [0.58, 0.58, 0.6],
            [0.58, 0.58, 0.6],
            [0.2, 0.2, 0.33],
            [0.58, 0.58, 0.6],
        ];

        const NC: usize = 9;
        const NSEG: usize = NC - 3;
        let first = (t * NSEG as Float).floor().min(1.0);
        t = t * NSEG as Float - first;

        let first = first as usize;

        let c0 = Spectrum::from_rgb(&c[first], SpectrumType::Reflectance);
        let c1 = Spectrum::from_rgb(&c[first + 1], SpectrumType::Reflectance);
        let c2 = Spectrum::from_rgb(&c[first + 2], SpectrumType::Reflectance);
        let c3 = Spectrum::from_rgb(&c[first + 3], SpectrumType::Reflectance);

        let mut s0 = c0 * (1.0 - t) + c1 * t;
        let mut s1 = c1 * (1.0 - t) + c2 * t;
        let s2 = c2 * (1.0 - t) + c3 * t;

        s0 = s0 * (1.0 - t) + s1 * t;
        s1 = s1 * (1.0 - t) + s2 * t;

        (s0 * (1.0 - t) + s1 * t) * 1.5
    }
}
