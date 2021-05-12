use crate::core::{
    interaction::SurfaceInteraction,
    material::{Material, TransportMode},
    pbrt::Float,
    spectrum::Spectrum,
    texture::TextureDt,
};
use derive_more::{Deref, DerefMut};
use std::any::Any;

#[derive(Debug)]
pub struct DisneyMaterial {
    color: TextureDt<Spectrum>,
    metallic: TextureDt<Float>,
    eta: TextureDt<Float>,
    roughness: TextureDt<Float>,
    specular_tint: TextureDt<Float>,
    anisotropic: TextureDt<Float>,
    sheen: TextureDt<Float>,
    sheen_tint: TextureDt<Float>,
    clear_coat: TextureDt<Float>,
    clear_coat_gloss: TextureDt<Float>,
    spec_trans: TextureDt<Float>,
    scatter_distance: TextureDt<Spectrum>,
    flatness: TextureDt<Float>,
    diff_trans: TextureDt<Float>,
    bump_map: TextureDt<Float>,
    thin: bool,
}

impl DisneyMaterial {
    fn new(
        color: TextureDt<Spectrum>,
        metallic: TextureDt<Float>,
        eta: TextureDt<Float>,
        roughness: TextureDt<Float>,
        specular_tint: TextureDt<Float>,
        anisotropic: TextureDt<Float>,
        sheen: TextureDt<Float>,
        sheen_tint: TextureDt<Float>,
        clear_coat: TextureDt<Float>,
        clear_coat_gloss: TextureDt<Float>,
        spec_trans: TextureDt<Float>,
        scatter_distance: TextureDt<Spectrum>,
        thin: bool,
        flatness: TextureDt<Float>,
        diff_trans: TextureDt<Float>,
        bump_map: TextureDt<Float>,
    ) -> Self {
        Self {
            color,
            metallic,
            eta,
            roughness,
            specular_tint,
            anisotropic,
            sheen,
            sheen_tint,
            clear_coat,
            clear_coat_gloss,
            spec_trans,
            scatter_distance,
            flatness,
            diff_trans,
            bump_map,
            thin,
        }
    }
}

impl Material for DisneyMaterial {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn compute_scattering_functions(
        &self,
        si: &mut SurfaceInteraction,
        mode: TransportMode,
        allow_multiple_lobes: bool,
    ) {
        todo!()
    }
}
