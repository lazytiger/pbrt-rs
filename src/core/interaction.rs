use crate::core::geometry::{offset_ray_origin, Normal3f, Point2f, Point3f, Ray, Vector3f};
use crate::core::medium::{MediumInterface, PhaseFunction};
use crate::core::pbrt::Float;
use crate::core::pbrt::SHADOW_EPSILON;
use crate::core::primitive::Primitive;
use crate::core::shape::Shape;
use crate::core::spectrum::Spectrum;
use std::ops::{Deref, DerefMut};

#[derive(Default, Clone)]
pub struct Interaction {
    pub p: Point3f,
    pub time: Float,
    pub error: Vector3f,
    pub wo: Vector3f,
    pub n: Normal3f,
    pub medium_interface: MediumInterface,
}

pub trait SpawnRayTo<T> {
    fn spawn_ray_to(&self, t: T) -> Ray;
}

impl Interaction {
    pub fn new(
        p: Point3f,
        n: Normal3f,
        time: Float,
        error: Vector3f,
        wo: Vector3f,
        medium_interface: MediumInterface,
    ) -> Self {
        Self {
            p,
            n,
            time,
            error,
            wo,
            medium_interface,
        }
    }

    pub fn is_surface_interaction(&self) -> bool {
        self.n != Normal3f::default()
    }

    pub fn is_medium_interaction(&self) -> bool {
        !self.is_surface_interaction()
    }

    pub fn spawn_ray(&self, d: &Vector3f) -> Ray {
        let origin = offset_ray_origin(&self.p, &self.error, &self.n, d);
        Ray::new(origin, *d, Float::INFINITY, self.time, None)
    }
}

impl SpawnRayTo<Point3f> for Interaction {
    fn spawn_ray_to(&self, p2: Point3f) -> Ray {
        let d = p2 - self.p;
        let origin = offset_ray_origin(&self.p, &self.error, &self.n, &d);
        Ray::new(origin, d, 1.0 - SHADOW_EPSILON, self.time, None)
    }
}

impl SpawnRayTo<&Interaction> for Interaction {
    fn spawn_ray_to(&self, it: &Interaction) -> Ray {
        let origin = offset_ray_origin(&self.p, &self.error, &self.n, &(it.p - self.p));
        let target = offset_ray_origin(&it.p, &it.error, &it.n, &(origin - it.p));
        let d = target - origin;
        Ray::new(origin, d, 1.0 - SHADOW_EPSILON, self.time, None)
    }
}

impl From<(Point3f, Vector3f, Float, MediumInterface)> for Interaction {
    fn from(data: (Point3f, Vector3f, Float, MediumInterface)) -> Self {
        let p = data.0;
        let wo = data.1;
        let time = data.2;
        let medium_interface = data.3;
        Self {
            p,
            wo,
            time,
            medium_interface,
            error: Default::default(),
            n: Default::default(),
        }
    }
}

impl From<(Point3f, Float, MediumInterface)> for Interaction {
    fn from(data: (Point3f, Float, MediumInterface)) -> Self {
        let p = data.0;
        let time = data.1;
        let medium_interface = data.2;
        Self {
            p,
            time,
            error: Default::default(),
            wo: Default::default(),
            n: Default::default(),
            medium_interface,
        }
    }
}

pub struct MediumInteraction<T: PhaseFunction> {
    base: Interaction,
    phase: T,
}

impl<T: PhaseFunction> MediumInteraction<T> {}

impl<T: PhaseFunction> Deref for MediumInteraction<T> {
    type Target = Interaction;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<T: PhaseFunction> DerefMut for MediumInteraction<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

#[derive(Copy, Clone, Default)]
pub struct Shading {
    pub n: Normal3f,
    dpdu: Vector3f,
    dpdv: Vector3f,
    dndu: Normal3f,
    dndv: Normal3f,
}

#[derive(Default)]
pub struct SurfaceInteraction<'a> {
    base: Interaction,
    uv: Point2f,
    pub(crate) dpdu: Vector3f,
    dpdv: Vector3f,
    dndu: Normal3f,
    dndv: Normal3f,
    shape: Option<&'a dyn Shape>,
    pub shading: Shading,
    pub primitive: Option<*const dyn Primitive>,
    dpdx: Vector3f,
    dpdy: Vector3f,
    dudx: Float,
    dvdx: Float,
    dudy: Float,
    dvdy: Float,
    face_index: i32,
}

impl<'a> Deref for SurfaceInteraction<'a> {
    type Target = Interaction;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<'a> DerefMut for SurfaceInteraction<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

impl<'a> SurfaceInteraction<'a> {
    pub fn new(
        p: Point3f,
        error: Vector3f,
        uv: Point2f,
        wo: Vector3f,
        dpdu: Vector3f,
        dpdv: Vector3f,
        dndu: Normal3f,
        dndv: Normal3f,
        time: Float,
        shape: Option<&'a dyn Shape>,
        face_index: i32,
    ) -> SurfaceInteraction {
        let mut si = SurfaceInteraction {
            base: Interaction::new(
                p,
                dpdu.cross(&dpdv).normalize(),
                time,
                error,
                wo,
                Default::default(),
            ),
            uv,
            dpdu,
            dpdv,
            dndu,
            dndv,
            shape,
            shading: Default::default(),
            primitive: None,
            dpdx: Default::default(),
            dpdy: Default::default(),
            dudx: 0.0,
            dvdx: 0.0,
            dudy: 0.0,
            dvdy: 0.0,
            face_index,
        };
        if let Some(shape) = &mut si.shape {
            if shape.reverse_orientation() ^ shape.transform_swap_handedness() {
                si.n *= -1.0;
                si.shading.n *= -1.0;
            }
        }
        si
    }

    pub fn set_shading_geometry(
        &mut self,
        dpdu: Vector3f,
        dpdv: Vector3f,
        dndu: Vector3f,
        dndv: Vector3f,
        orientation_is_authoritative: bool,
    ) {
        self.shading.n = dpdu.cross(&dpdv).normalize();
        if orientation_is_authoritative {
            self.n = self.n.face_forward(self.shading.n);
        } else {
            self.shading.n = self.shading.n.face_forward(self.n);
        }

        self.shading.dpdu = dpdu;
        self.shading.dpdv = dpdv;
        self.shading.dndu = dndu;
        self.shading.dndv = dndv;
    }

    pub fn le(&self, w: &Vector3f) -> Spectrum {
        if let Some(primitive) = self.primitive {
            let primitive = unsafe { &*primitive };
            let area = primitive.get_area_light();
            if let Some(area) = area {
                return area.l(self, w);
            }
        }
        Spectrum::new(0.0)
    }
}
