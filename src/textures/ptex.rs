use crate::core::{interaction::SurfaceInteraction, pbrt::Float, texture::Texture};
use derive_more::{Deref, DerefMut};
use std::any::Any;

pub struct PtexTexture {
    valid: bool,
    filename: String,
    gamma: Float,
}

impl PtexTexture {
    pub fn new(filename: String, gamma: Float) -> Self {
        Self {
            valid: false,
            filename,
            gamma,
        }
    }
}

impl<T> Texture<T> for PtexTexture
where
    T: 'static,
    T: Default,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, si: &SurfaceInteraction) -> T {
        if !self.valid {
            return T::default();
        }

        todo!()
    }
}
