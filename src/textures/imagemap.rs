use crate::core::{
    geometry::Vector2f,
    interaction::SurfaceInteraction,
    lightdistrib::SpatialLightDistribution,
    mipmap::{ClampLerp, ImageWrap, MIPMap},
    pbrt::{float_to_bits, inverse_gamma_correct, Float},
    spectrum::{RGBSpectrum, Spectrum, SpectrumType},
    texture::{Texture, TextureMapping2DDt},
};
use derive_more::{Deref, DerefMut};
use std::{
    any::Any,
    cmp::Ordering,
    collections::HashMap,
    hash::{Hash, Hasher},
    ops::{Add, AddAssign, Div, Mul},
    sync::{Arc, RwLock},
};

#[derive(PartialEq, PartialOrd)]
struct TexInfo {
    filename: String,
    do_trilinear: bool,
    max_aniso: Float,
    scale: Float,
    gamma: bool,
    wrap_mode: ImageWrap,
}

impl Hash for TexInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u8(if self.do_trilinear { 1 } else { 0 });
        state.write_u32(float_to_bits(self.max_aniso));
        state.write_u32(float_to_bits(self.max_aniso));
        state.write_u8(if self.gamma { 1 } else { 0 });
        state.write_u8(match self.wrap_mode {
            ImageWrap::Repeat => 0,
            ImageWrap::Black => 1,
            ImageWrap::Clamp => 2,
        });
        state.write(self.filename.as_bytes());
    }
}

impl Eq for TexInfo {}

impl TexInfo {
    pub fn new(f: String, dt: bool, ma: Float, wm: ImageWrap, sc: Float, gamma: bool) -> Self {
        Self {
            filename: f,
            do_trilinear: dt,
            max_aniso: ma,
            wrap_mode: wm,
            scale: sc,
            gamma,
        }
    }
}

pub struct ImageTexture {
    mapping: TextureMapping2DDt,
    mipmap: Arc<MIPMap<RGBSpectrum>>,
}

lazy_static::lazy_static! {
    static ref TEXTURES: RwLock<HashMap<TexInfo, Arc<MIPMap<RGBSpectrum>>>> = RwLock::new(HashMap::new());
}

impl ImageTexture {
    pub fn new(
        m: TextureMapping2DDt,
        filename: String,
        do_tri: bool,
        max_aniso: Float,
        wm: ImageWrap,
        scale: Float,
        gamma: bool,
    ) -> Self {
        Self {
            mapping: m,
            mipmap: Self::get_texture(filename, do_tri, max_aniso, wm, scale, gamma),
        }
    }

    pub fn clear_cache() {
        TEXTURES.write().unwrap().clear();
    }

    fn get_texture(
        filename: String,
        do_trilinear: bool,
        max_aniso: Float,
        wm: ImageWrap,
        scale: Float,
        gamma: bool,
    ) -> Arc<MIPMap<RGBSpectrum>> {
        let text_info = TexInfo::new(filename, do_trilinear, max_aniso, wm, scale, gamma);
        if let Some(mm) = TEXTURES.read().unwrap().get(&text_info) {
            return mm.clone();
        }
        todo!()
    }

    fn convert_in(from: &RGBSpectrum, to: &mut RGBSpectrum, scale: Float, gamma: bool) {
        for i in 0..RGBSpectrum::N {
            (*to)[i] = scale * {
                if gamma {
                    inverse_gamma_correct(from[i])
                } else {
                    from[i]
                }
            };
        }
    }

    fn convert_in_float(from: &RGBSpectrum, to: &mut Float, scale: Float, gamma: bool) {
        *to = scale
            * if gamma {
                inverse_gamma_correct(from.y_value())
            } else {
                from.y_value()
            }
    }

    fn convert_out(from: &RGBSpectrum, to: &mut Spectrum) {
        let mut rgb = [0.0; 3];
        from.to_rgb(&mut rgb);
        *to = Spectrum::from_rgb(&rgb, SpectrumType::Reflectance);
    }

    fn convert_out_float(from: Float, to: &mut Float) {
        *to = from;
    }
}

impl Texture<Spectrum> for ImageTexture {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, si: &SurfaceInteraction) -> RGBSpectrum {
        let mut dstdx = Vector2f::default();
        let mut dstdy = Vector2f::default();
        let st = self.mapping.map(si, &mut dstdx, &mut dstdy);
        let mem = self.mipmap.lookup2(&st, dstdx, dstdy);
        let mut ret = Spectrum::default();
        Self::convert_out(&mem, &mut ret);
        ret
    }
}
