use crate::core::geometry::{Normal3f, Point2f, Point3f, Vector3f};
use crate::core::medium::{MediumInterface, PhaseFunction};
use crate::core::primitive::Primitive;
use crate::core::shape::Shape;
use crate::inherit;
use crate::Float;
use std::ops::Deref;

pub struct Interaction {
    p: Point3f,
    time: Float,
    error: Vector3f,
    wo: Vector3f,
    medium_interface: MediumInterface,
}

impl Interaction {
    pub fn new(
        p: Point3f,
        time: Float,
        error: Vector3f,
        wo: Vector3f,
        medium_interface: MediumInterface,
    ) -> Self {
        Self {
            p,
            time,
            error,
            wo,
            medium_interface,
        }
    }
}

pub struct MediumInteraction<T: PhaseFunction> {
    base: Interaction,
    phase: T,
}

impl<T: PhaseFunction> MediumInteraction<T> {}

inherit!(MediumInteraction, Interaction, base, PhaseFunction);

struct Shading {
    n: Normal3f,
    dpdu: Vector3f,
    dpdv: Vector3f,
    dndu: Normal3f,
    dndv: Normal3f,
}

pub struct SurfaceInteraction<T: Shape, P: Primitive> {
    base: Interaction,
    uv: Point2f,
    dpdu: Vector3f,
    dpdv: Vector3f,
    dndu: Normal3f,
    dndv: Normal3f,
    shape: T,
    shading: Shading,
    primitive: P,
}

impl<T: Shape, P: Primitive> Deref for SurfaceInteraction<T, P> {
    type Target = Interaction;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}
