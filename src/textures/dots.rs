use crate::core::{
    geometry::{Point2f, Vector2f},
    interaction::SurfaceInteraction,
    texture::{noise, Texture, TextureDt, TextureMapping2DDt},
};
use derive_more::{Deref, DerefMut};
use std::{any::Any, cmp::max};

pub struct DotsTexture<T> {
    mapping: TextureMapping2DDt,
    outside_dot: TextureDt<T>,
    inside_dot: TextureDt<T>,
}

impl<T> DotsTexture<T> {
    pub fn new(
        mapping: TextureMapping2DDt,
        outside_dot: TextureDt<T>,
        inside_dot: TextureDt<T>,
    ) -> Self {
        Self {
            mapping,
            outside_dot,
            inside_dot,
        }
    }
}

impl<T> Texture<T> for DotsTexture<T>
where
    T: 'static,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, si: &SurfaceInteraction) -> T {
        let mut dstdx = Vector2f::default();
        let mut dstdy = Vector2f::default();
        let st = self.mapping.map(si, &mut dstdx, &mut dstdy);
        let s_cell = (st[0] + 0.5).floor();
        let t_cell = (st[1] + 0.5).floor();
        if noise(s_cell + 0.5, t_cell + 0.5, 0.5) != 0.0 {
            let radius = 0.35;
            let max_shift = 0.5 - radius;
            let s_center = s_cell + max_shift * noise(s_cell + 1.5, t_cell + 2.8, 0.5);
            let t_center = t_cell + max_shift * noise(t_cell + 4.5, t_cell + 9.8, 0.5);
            let dst = st - Point2f::new(s_center, t_center);
            if dst.length_squared() < radius * radius {
                return self.inside_dot.evaluate(si);
            }
        }
        self.outside_dot.evaluate(si)
    }
}
