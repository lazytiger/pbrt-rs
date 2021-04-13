use crate::core::{
    bssrdf::BSSRDFDt,
    geometry::{offset_ray_origin, Normal3f, Point2f, Point3f, Ray, Vector3f},
    medium::{MediumInterface, PhaseFunction},
    pbrt::{Float, SHADOW_EPSILON},
    primitive::{Primitive, PrimitiveDt},
    reflection::BSDF,
    shape::{Shape, ShapeDt},
    spectrum::Spectrum,
};
use derive_more::{Deref, DerefMut};
use std::{
    any::Any,
    sync::{Arc, Mutex, RwLock},
};

pub trait Interaction {
    fn as_any(&self) -> &dyn Any;
    fn as_base(&self) -> &BaseInteraction;
    fn as_base_mut(&mut self) -> &mut BaseInteraction;
    fn is_surface_interaction(&self) -> bool;
    fn is_medium_interaction(&self) -> bool;
}

impl Interaction for BaseInteraction {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_base(&self) -> &BaseInteraction {
        self
    }

    fn as_base_mut(&mut self) -> &mut BaseInteraction {
        self
    }

    fn is_surface_interaction(&self) -> bool {
        false
    }

    fn is_medium_interaction(&self) -> bool {
        false
    }
}

impl<T: 'static + PhaseFunction> Interaction for MediumInteraction<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_base(&self) -> &BaseInteraction {
        &self.base
    }

    fn as_base_mut(&mut self) -> &mut BaseInteraction {
        &mut self.base
    }

    fn is_surface_interaction(&self) -> bool {
        false
    }

    fn is_medium_interaction(&self) -> bool {
        true
    }
}

impl Interaction for SurfaceInteraction {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_base(&self) -> &BaseInteraction {
        &self.base
    }

    fn as_base_mut(&mut self) -> &mut BaseInteraction {
        &mut self.base
    }

    fn is_surface_interaction(&self) -> bool {
        true
    }

    fn is_medium_interaction(&self) -> bool {
        false
    }
}

pub type InteractionDt = Arc<Box<dyn Interaction>>;
pub type InteractionDtMut = Arc<Mutex<Box<dyn Interaction>>>;
pub type InteractionDtRw = Arc<RwLock<Box<dyn Interaction>>>;

#[derive(Default, Clone)]
pub struct BaseInteraction {
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

impl BaseInteraction {
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

    pub fn spawn_ray(&self, d: &Vector3f) -> Ray {
        let origin = offset_ray_origin(&self.p, &self.error, &self.n, d);
        Ray::new(origin, *d, Float::INFINITY, self.time, None)
    }
}

impl SpawnRayTo<Point3f> for BaseInteraction {
    fn spawn_ray_to(&self, p2: Point3f) -> Ray {
        let d = p2 - self.p;
        let origin = offset_ray_origin(&self.p, &self.error, &self.n, &d);
        Ray::new(origin, d, 1.0 - SHADOW_EPSILON, self.time, None)
    }
}

impl SpawnRayTo<&BaseInteraction> for BaseInteraction {
    fn spawn_ray_to(&self, it: &BaseInteraction) -> Ray {
        let origin = offset_ray_origin(&self.p, &self.error, &self.n, &(it.p - self.p));
        let target = offset_ray_origin(&it.p, &it.error, &it.n, &(origin - it.p));
        let d = target - origin;
        Ray::new(origin, d, 1.0 - SHADOW_EPSILON, self.time, None)
    }
}

impl From<(Point3f, Vector3f, Float, MediumInterface)> for BaseInteraction {
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

impl From<(Point3f, Float, MediumInterface)> for BaseInteraction {
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

#[derive(Deref, DerefMut)]
pub struct MediumInteraction<T: PhaseFunction> {
    #[deref]
    #[deref_mut]
    base: BaseInteraction,
    phase: T,
}

impl<T: PhaseFunction> MediumInteraction<T> {}

#[derive(Copy, Clone, Default)]
pub struct Shading {
    pub n: Normal3f,
    pub(crate) dpdu: Vector3f,
    dpdv: Vector3f,
    dndu: Normal3f,
    dndv: Normal3f,
}

#[derive(Default, Deref, DerefMut)]
pub struct SurfaceInteraction {
    #[deref]
    #[deref_mut]
    base: BaseInteraction,
    uv: Point2f,
    pub(crate) dpdu: Vector3f,
    dpdv: Vector3f,
    dndu: Normal3f,
    dndv: Normal3f,
    shape: Option<ShapeDt>,
    pub shading: Shading,
    pub primitive: Option<PrimitiveDt>,
    pub bsdf: Option<Arc<BSDF>>,
    pub bssrdf: Option<BSSRDFDt>,
    dpdx: Vector3f,
    dpdy: Vector3f,
    dudx: Float,
    dvdx: Float,
    dudy: Float,
    dvdy: Float,
    face_index: i32,
}

impl SurfaceInteraction {
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
        shape: Option<ShapeDt>,
        face_index: i32,
    ) -> SurfaceInteraction {
        let mut si = SurfaceInteraction {
            base: BaseInteraction::new(
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
            bsdf: None,
            bssrdf: None,
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
        if let Some(primitive) = &self.primitive {
            let area = primitive.get_area_light();
            if let Some(area) = area {
                return area.l(self, w);
            }
        }
        Spectrum::new(0.0)
    }
}
