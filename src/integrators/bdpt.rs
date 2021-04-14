use crate::{
    core::{
        camera::CameraDt,
        geometry::{Bounds2i, Normal3f, Point2f, Point3f, Ray, RayDifferentials, Vector3f},
        integrator::{BaseSamplerIntegrator, Integrator},
        interaction::{
            BaseInteraction, Interaction, InteractionDt, MediumInteraction, SurfaceInteraction,
        },
        light::{is_delta_light, LightDt, LightFlags},
        material::TransportMode,
        pbrt::{any_equal, Float, PI},
        reflection::BxDFType,
        sampler::SamplerDtRw,
        sampling::Distribution1D,
        scene::Scene,
        spectrum::Spectrum,
    },
    shapes::curve::CurveType::Flat,
};
use derive_more::{Deref, DerefMut};
use std::{
    any::Any,
    collections::HashMap,
    hash::{Hash, Hasher},
    io::SeekFrom::End,
    path::Prefix::Verbatim,
    raw::TraitObject,
    sync::Arc,
};

pub fn correct_shading_normal(
    isect: &SurfaceInteraction,
    wo: &Vector3f,
    wi: &Vector3f,
    mode: TransportMode,
) -> Float {
    todo!()
}

#[derive(Deref, DerefMut, Default)]
pub struct EndPointInteraction {
    #[deref]
    #[deref_mut]
    base: BaseInteraction,
    camera: Option<CameraDt>,
    light: Option<LightDt>,
}

impl From<(BaseInteraction, CameraDt)> for EndPointInteraction {
    fn from(input: (BaseInteraction, CameraDt)) -> Self {
        let it = input.0;
        let camera = input.1;
        Self {
            base: it,
            camera: Some(camera),
            light: None,
        }
    }
}

impl From<(CameraDt, &Ray)> for EndPointInteraction {
    fn from(input: (CameraDt, &Ray)) -> Self {
        let camera = input.0;
        let ray = input.1;
        Self {
            base: BaseInteraction::from((ray.o, ray.time, ray.medium.clone().into())),
            camera: Some(camera),
            light: None,
        }
    }
}

impl From<(LightDt, &Ray, Normal3f)> for EndPointInteraction {
    fn from(input: (LightDt, &Ray, Normal3f)) -> Self {
        let light = input.0;
        let r = input.1;
        let nl = input.2;
        let mut epi = Self {
            base: BaseInteraction::from((r.o, r.time, r.medium.clone().into())),
            camera: None,
            light: Some(light),
        };
        epi.n = nl;
        epi
    }
}

impl From<(BaseInteraction, LightDt)> for EndPointInteraction {
    fn from(input: (BaseInteraction, LightDt)) -> Self {
        let it = input.0;
        let light = input.1;
        Self {
            base: it,
            light: Some(light),
            camera: None,
        }
    }
}

impl From<&Ray> for EndPointInteraction {
    fn from(ray: &Ray) -> Self {
        let mut epi = Self {
            base: BaseInteraction::from((ray.point(1.0), ray.time, ray.medium.clone().into())),
            light: None,
            camera: None,
        };
        epi.n = -ray.d;
        epi
    }
}

impl Interaction for EndPointInteraction {
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
        false
    }
}

pub enum VertexType {
    Camera,
    Light,
    Surface,
    Medium,
}

impl Default for VertexType {
    fn default() -> Self {
        VertexType::Camera
    }
}

#[derive(Default)]
pub struct ScopedAssignment<'a, T: Default + Copy> {
    target: Option<&'a mut T>,
    backup: T,
}

impl<'a, T: Default + Copy> Drop for ScopedAssignment<'a, T> {
    fn drop(&mut self) {
        if self.target.is_some() {
            *self.target.take().unwrap() = self.backup;
        }
    }
}

impl<'a, T: Default + Copy> ScopedAssignment<'a, T> {
    pub fn new(mut target: Option<&'a mut T>, value: T) -> Self {
        let mut backup = value;
        if target.is_some() {
            let p_target = target.take().unwrap();
            backup = *p_target;
            *p_target = value;
            target = Some(p_target);
        }
        Self { target, backup }
    }

    pub fn replace(&mut self, other: &mut ScopedAssignment<'a, T>) {
        if self.target.is_some() {
            *self.target.take().unwrap() = self.backup;
        }
        self.target = other.target.take();
        self.backup = other.backup;
    }
}

pub struct LightKey(LightDt);

impl PartialEq for LightKey {
    fn eq(&self, other: &Self) -> bool {
        any_equal(self.0.as_any(), other.0.as_any())
    }
}

impl Eq for LightKey {}

impl Hash for LightKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        unsafe {
            let to: TraitObject = unsafe { std::mem::transmute(self.0.as_any()) };
            to.data.hash(state);
        }
    }
}

#[inline]
pub fn infinite_light_density(
    scene: &Scene,
    light_distr: &Distribution1D,
    light2distr_index: &HashMap<LightKey, usize>,
    w: &Vector3f,
) -> Float {
    let mut pdf = 0.0;
    for light in &scene.infinite_lights {
        let index = *light2distr_index.get(&LightKey(light.clone())).unwrap();
        pdf += light.pdf_li(Arc::new(Box::new(BaseInteraction::default())), &-*w)
            * light_distr.func[index];
    }
    pdf / (light_distr.func_int as Float * light_distr.count() as Float)
}

pub struct BDPTIntegrator {
    sampler: SamplerDtRw,
    camera: CameraDt,
    max_depth: usize,
    visualize_strategies: bool,
    visualize_weights: bool,
    pixel_bounds: Bounds2i,
}

impl BDPTIntegrator {
    pub fn new(
        sampler: SamplerDtRw,
        camera: CameraDt,
        max_depth: usize,
        visualize_strategies: bool,
        visualize_weights: bool,
        pixel_bounds: Bounds2i,
    ) -> Self {
        Self {
            sampler,
            camera,
            max_depth,
            visualize_strategies,
            visualize_weights,
            pixel_bounds,
        }
    }
}

impl Integrator for BDPTIntegrator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn render(&self, scene: &Scene) {
        todo!()
    }

    fn li(
        &self,
        ray: &RayDifferentials,
        scene: &Scene,
        sampler: SamplerDtRw,
        depth: i32,
    ) -> Spectrum {
        unimplemented!()
    }
}

#[derive(Default)]
pub struct Vertex {
    typ: VertexType,
    beta: Spectrum,
    ei: EndPointInteraction,
    mi: MediumInteraction,
    si: SurfaceInteraction,
    delta: bool,
    pdf_fwd: Float,
    pdf_rev: Float,
}

impl Vertex {
    pub fn new(typ: VertexType, ei: EndPointInteraction, beta: Spectrum) -> Self {
        Self {
            typ,
            ei,
            mi: Default::default(),
            si: Default::default(),
            delta: false,
            pdf_fwd: 0.0,
            beta,
            pdf_rev: 0.0,
        }
    }

    pub fn create_camera(camera: CameraDt, ray: &Ray, beta: Spectrum) -> Self {
        Vertex::new(
            VertexType::Camera,
            EndPointInteraction::from((camera, ray)),
            beta,
        )
    }

    pub fn create_camera2(camera: CameraDt, it: BaseInteraction, beta: Spectrum) -> Self {
        Vertex::new(
            VertexType::Camera,
            EndPointInteraction::from((it, camera)),
            beta,
        )
    }

    pub fn create_light(
        light: LightDt,
        ray: &Ray,
        n_light: Normal3f,
        le: Spectrum,
        pdf: Float,
    ) -> Self {
        let mut v = Vertex::new(
            VertexType::Light,
            EndPointInteraction::from((light, ray, n_light)),
            le,
        );
        v.pdf_fwd = pdf;
        v
    }

    pub fn create_light2(ei: EndPointInteraction, beta: Spectrum, pdf: Float) -> Self {
        let mut v = Vertex::new(VertexType::Light, ei, beta);
        v.pdf_fwd = pdf;
        v
    }

    pub fn create_medium(mi: MediumInteraction, beta: Spectrum, pdf: Float, prev: Vertex) -> Self {
        let mut v = Vertex::from((mi, beta));
        v.pdf_fwd = prev.convert_density(pdf, &v);
        v
    }

    pub fn create_surface(
        si: SurfaceInteraction,
        beta: Spectrum,
        pdf: Float,
        prev: Vertex,
    ) -> Self {
        let mut v = Vertex::from((si, beta));
        v.pdf_fwd = prev.convert_density(pdf, &v);
        v
    }

    pub fn get_interaction(&self) -> &dyn Interaction {
        match self.typ {
            VertexType::Medium => &self.mi,
            VertexType::Surface => &self.si,
            _ => &self.ei,
        }
    }

    pub fn p(&self) -> &Point3f {
        &self.get_interaction().as_base().p
    }

    pub fn time(&self) -> Float {
        self.get_interaction().as_base().time
    }

    pub fn ng(&self) -> &Normal3f {
        &self.get_interaction().as_base().n
    }

    pub fn ns(&self) -> &Normal3f {
        if let VertexType::Surface = self.typ {
            &self.si.shading.n
        } else {
            self.ng()
        }
    }

    pub fn is_on_surface(&self) -> bool {
        *self.ng() != Normal3f::default()
    }

    pub fn f(&self, next: &Vertex, mode: TransportMode) -> Spectrum {
        let wi = *next.p() - *self.p();
        if wi.length_squared() == 0.0 {
            return Spectrum::new(0.0);
        }
        let wi = wi.normalize();
        match self.typ {
            VertexType::Surface => {
                self.si
                    .bsdf
                    .clone()
                    .unwrap()
                    .f(&self.si.wo, &wi, BxDFType::all())
                    * correct_shading_normal(&self.si, &self.si.wo, &wi, mode)
            }
            VertexType::Medium => Spectrum::new(self.mi.phase.p(&self.mi.wo, &wi)),
            _ => Spectrum::new(0.0),
        }
    }

    pub fn is_connectible(&self) -> bool {
        match self.typ {
            VertexType::Medium => true,
            VertexType::Light => {
                (self.ei.light.clone().unwrap().flags() & LightFlags::DELTA_DIRECTION).is_empty()
            }
            VertexType::Camera => true,
            VertexType::Surface => {
                self.si
                    .bsdf
                    .clone()
                    .unwrap()
                    .num_components(BxDFType::all() - BxDFType::BSDF_SPECULAR)
                    > 0
            }
        }
    }

    pub fn is_light(&self) -> bool {
        match self.typ {
            VertexType::Camera => false,
            VertexType::Light => true,
            VertexType::Surface => self
                .si
                .primitive
                .clone()
                .unwrap()
                .get_area_light()
                .is_some(),
            VertexType::Medium => false,
        }
    }

    pub fn is_delta_light(&self) -> bool {
        if let VertexType::Light = self.typ {
            if let Some(light) = &self.ei.light {
                return is_delta_light(light.flags());
            }
        }
        false
    }

    pub fn is_infinite_light(&self) -> bool {
        if let VertexType::Light = self.typ {
            if let Some(light) = &self.ei.light {
                let flags = light.flags();
                if !(flags & LightFlags::INFINITE).is_empty()
                    || !(flags & LightFlags::DELTA_DIRECTION).is_empty()
                {
                    return true;
                } else {
                    return false;
                }
            } else {
                return true;
            }
        }
        false
    }
    pub fn le(&self, scene: &Scene, v: &Vertex) -> Spectrum {
        if !self.is_light() {
            return Spectrum::new(0.0);
        }

        let w = *v.p() - *self.p();
        if w.length_squared() == 0.0 {
            return Spectrum::new(0.0);
        }

        let w = w.normalize();
        if self.is_infinite_light() {
            let mut le = Spectrum::new(0.0);
            for light in &scene.infinite_lights {
                le += light.le(&Ray::new(*self.p(), -w, Float::INFINITY, 0.0, None).into());
            }
            le
        } else {
            let light = self.si.primitive.clone().unwrap().get_area_light().unwrap();
            light.l(&self.si, &w)
        }
    }

    pub fn convert_density(&self, mut pdf: Float, next: &Vertex) -> Float {
        if next.is_infinite_light() {
            return pdf;
        }
        let w = *next.p() - *self.p();
        if w.length_squared() == 0.0 {
            return 0.0;
        }

        let inv_dist2 = 1.0 / w.length_squared();
        if next.is_on_surface() {
            pdf *= next.ng().abs_dot(&(w * inv_dist2.sqrt()));
        }
        pdf * inv_dist2
    }

    pub fn pdf(&self, scene: &Scene, prev: Option<&Vertex>, next: &Vertex) -> Float {
        if let VertexType::Light = self.typ {
            return self.pdf_light(scene, next);
        }

        let wn = *next.p() - *self.p();
        if wn.length_squared() == 0.0 {
            return 0.0;
        }
        let wn = wn.normalize();
        let mut wp = Vector3f::default();
        if let Some(prev) = prev {
            wp = *prev.p() - *self.p();
            if wp.length_squared() == 0.0 {
                return 0.0;
            }
            wp = wp.normalize();
        }

        let mut pdf = 0.0;
        let mut unused = 0.0;
        match self.typ {
            VertexType::Camera => self.ei.camera.clone().unwrap().pdf_we(
                &self.ei.spawn_ray(&wn),
                &mut unused,
                &mut pdf,
            ),
            VertexType::Light => {
                panic!("vertex::pdf unimplemented")
            }
            VertexType::Surface => {
                pdf = self.si.bsdf.clone().unwrap().pdf(&wp, &wn, BxDFType::all());
            }
            VertexType::Medium => {
                pdf = self.mi.phase.p(&wp, &wn);
            }
        }

        self.convert_density(pdf, next)
    }

    pub fn pdf_light(&self, scene: &Scene, v: &Vertex) -> Float {
        let mut w = *v.p() - *self.p();
        let inv_dist2 = 1.0 / w.length_squared();
        w *= inv_dist2.sqrt();
        let mut pdf = 0.0;
        if self.is_infinite_light() {
            let mut world_center = Point3f::default();
            let mut world_radius = 0.0;
            scene
                .world_bound()
                .bounding_sphere(&mut world_center, &mut world_radius);
            pdf = 1.0 / (PI * world_radius * world_radius);
        } else {
            let mut pdf_pos = 0.0;
            let mut pdf_dir = 0.0;
            let ray = Ray::new(*self.p(), w, Float::INFINITY, self.time(), None);
            let light = if let VertexType::Light = self.typ {
                self.ei
                    .light
                    .clone()
                    .unwrap()
                    .pdf_le(&ray, self.ng(), &mut pdf_pos, &mut pdf_dir);
            } else {
                self.si
                    .primitive
                    .clone()
                    .unwrap()
                    .get_area_light()
                    .unwrap()
                    .pdf_le(&ray, self.ng(), &mut pdf_pos, &mut pdf_dir);
            };
            pdf = pdf_dir * inv_dist2;
        }
        if v.is_on_surface() {
            pdf *= v.ng().abs_dot(&w);
        }
        pdf
    }

    pub fn pdf_light_origin(
        &self,
        scene: &Scene,
        v: &Vertex,
        light_distr: &Distribution1D,
        light2distr_index: &HashMap<LightKey, usize>,
    ) -> Float {
        let w = *v.p() - *self.p();
        if w.length_squared() == 0.0 {
            return 0.0;
        }
        let w = w.normalize();
        if self.is_infinite_light() {
            infinite_light_density(scene, light_distr, light2distr_index, &w)
        } else {
            let (mut pdf_pos, mut pdf_dir, mut pdf_choice) = (0.0, 0.0, 0.0);
            let light = if let VertexType::Light = self.typ {
                self.ei.light.clone().unwrap()
            } else {
                self.si.primitive.clone().unwrap().get_area_light().unwrap()
            };
            let index = *light2distr_index.get(&LightKey(light.clone())).unwrap();
            pdf_choice = light_distr.discrete_pdf(index);
            light.pdf_le(
                &Ray::new(*self.p(), w, Float::INFINITY, self.time(), None),
                self.ng(),
                &mut pdf_pos,
                &mut pdf_dir,
            );
            pdf_pos * pdf_choice
        }
    }
}

impl From<(MediumInteraction, Spectrum)> for Vertex {
    fn from((mi, beta): (MediumInteraction, Spectrum)) -> Self {
        Self {
            mi,
            si: Default::default(),
            delta: false,
            pdf_fwd: 0.0,
            beta,
            typ: VertexType::Medium,
            ei: Default::default(),
            pdf_rev: 0.0,
        }
    }
}
impl From<(SurfaceInteraction, Spectrum)> for Vertex {
    fn from((si, beta): (SurfaceInteraction, Spectrum)) -> Self {
        Self {
            typ: VertexType::Surface,
            ei: EndPointInteraction::default(),
            si,
            mi: MediumInteraction::default(),
            beta,
            pdf_fwd: 0.0,
            pdf_rev: 0.0,
            delta: false,
        }
    }
}

fn generate_camera_sub_path(
    scene: &Scene,
    sampler: SamplerDtRw,
    max_depth: usize,
    camera: CameraDt,
    p_film: Point2f,
    path: &mut [Vertex],
) {
    todo!()
}

fn generate_light_sub_path(
    scene: &Scene,
    sampler: &SamplerDtRw,
    max_depth: usize,
    time: Float,
    light_distr: &Distribution1D,
    light2index: &HashMap<LightKey, usize>,
    path: &mut [Vertex],
) {
    todo!()
}

fn connect_bdpt(
    scene: &Scene,
    light_vertices: &mut [Vertex],
    camera_vertices: &mut [Vertex],
    s: usize,
    light_distr: &Distribution1D,
    light2index: &HashMap<LightKey, usize>,
    camera: CameraDt,
    sampler: SamplerDtRw,
    p_raster: &mut Point2f,
    mis_weight: Option<&mut Float>,
) {
    todo!()
}
