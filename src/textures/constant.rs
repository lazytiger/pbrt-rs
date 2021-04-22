use crate::core::{interaction::SurfaceInteraction, texture::Texture};
use derive_more::{Deref, DerefMut};
use std::any::Any;

pub struct ConstantTexture<T> {
    value: T,
}

impl<T> ConstantTexture<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T> Texture<T> for ConstantTexture<T>
where
    T: 'static,
    T: Copy,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn evaluate(&self, si: &SurfaceInteraction) -> T {
        self.value
    }
}
