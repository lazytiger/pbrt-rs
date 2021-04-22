use crate::core::{
    geometry::{Vector2f, Vector3f},
    interaction::SurfaceInteraction,
    pbrt::Float,
    texture::{Texture, TextureDt, TextureMapping2DDt, TextureMapping3DDt},
};
use derive_more::{Deref, DerefMut};
use std::{
    any::Any,
    ops::{Add, Mul},
};

pub enum AAMethod {
    None,
    ClosedForm,
}

pub struct Checkerboard2DTexture<T> {
    mapping: TextureMapping2DDt,
    tex1: TextureDt<T>,
    tex2: TextureDt<T>,
    aa_method: AAMethod,
}

impl<T> Checkerboard2DTexture<T> {
    pub fn new(
        mapping: TextureMapping2DDt,
        tex1: TextureDt<T>,
        tex2: TextureDt<T>,
        aa_method: AAMethod,
    ) -> Self {
        Self {
            mapping,
            tex1,
            tex2,
            aa_method,
        }
    }
}

impl<T> Texture<T> for Checkerboard2DTexture<T>
where
    T: 'static,
    T: Mul<Float, Output = T>,
    T: Add<Output = T>,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, si: &SurfaceInteraction) -> T {
        let mut dstdx = Vector2f::default();
        let mut dstdy = Vector2f::default();
        let st = self.mapping.map(si, &mut dstdx, &mut dstdy);
        match self.aa_method {
            AAMethod::None => {
                if (st[0].floor() as i32 + st[1].floor() as i32) % 2 == 0 {
                    self.tex1.evaluate(si)
                } else {
                    self.tex2.evaluate(si)
                }
            }
            AAMethod::ClosedForm => {
                let ds = dstdx[0].abs().max(dstdy[0].abs());
                let dt = dstdx[1].abs().max(dstdy[1].abs());
                let s0 = st[0] - ds;
                let s1 = st[0] + ds;
                let t0 = st[1] - dt;
                let t1 = st[1] + dt;
                if s0.floor() == s1.floor() && t0.floor() == t1.floor() {
                    if (st[0].floor() as i32 + st[1].floor() as i32) % 2 == 0 {
                        self.tex1.evaluate(si)
                    } else {
                        self.tex2.evaluate(si)
                    }
                } else {
                    let bump_int = |x: Float| {
                        (x / 2.0).floor() + 2.0 * (x / 2.0 - (x / 2.0).floor() - 0.5).max(0.0)
                    };
                    let sint = (bump_int(s1) - bump_int(s0)) / (2.0 * ds);
                    let tint = (bump_int(t1) - bump_int(t0)) / (2.0 * dt);
                    let mut area2 = sint + tint - 2.0 * sint * tint;
                    if ds > 1.0 || dt > 1.0 {
                        area2 = 0.5;
                    }
                    self.tex1.evaluate(si) * (1.0 - area2) + self.tex2.evaluate(si) * area2
                }
            }
        }
    }
}

pub struct Checkerboard3DTexture<T> {
    mapping: TextureMapping3DDt,
    tex1: TextureDt<T>,
    tex2: TextureDt<T>,
}

impl<T> Checkerboard3DTexture<T> {
    pub fn new(mapping: TextureMapping3DDt, tex1: TextureDt<T>, tex2: TextureDt<T>) -> Self {
        Self {
            mapping,
            tex1,
            tex2,
        }
    }
}

impl<T> Texture<T> for Checkerboard3DTexture<T>
where
    T: 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, si: &SurfaceInteraction) -> T {
        let mut dpdx = Vector3f::default();
        let mut dpdy = Vector3f::default();
        let p = self.mapping.map(si, &mut dpdx, &mut dpdy);
        if (p.x.floor() as i32 + p.y.floor() as i32 + p.z.floor() as i32) % 2 == 0 {
            self.tex1.evaluate(si)
        } else {
            self.tex2.evaluate(si)
        }
    }
}
