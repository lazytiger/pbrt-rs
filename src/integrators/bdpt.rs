use crate::{
    core::{
        camera::{CameraDt, CameraSample},
        film::Film,
        geometry::{
            Bounds2f, Bounds2i, Normal3f, Point2f, Point2i, Point3f, Ray, RayDifferentials,
            Vector3f,
        },
        integrator::{Integrator, SamplerIntegrator},
        interaction::{
            BaseInteraction, Interaction, InteractionDt, MediumInteraction, SurfaceInteraction,
        },
        light::{is_delta_light, LightDt, LightFlags, VisibilityTester},
        lightdistrib::create_light_sample_distribution,
        material::TransportMode,
        pbrt::{any_equal, Float, PI},
        reflection::BxDFType,
        sampler::SamplerDtRw,
        sampling::Distribution1D,
        scene::Scene,
        spectrum::Spectrum,
    },
    filters::boxf::create_box_filter,
    integrators::bdpt::VertexType::Surface,
    parallel_for_2d,
    shapes::curve::CurveType::Flat,
};
use derive_more::{Deref, DerefMut};
use std::{
    any::Any,
    cmp::max,
    collections::HashMap,
    hash::{Hash, Hasher},
    io::SeekFrom::End,
    path::Prefix::Verbatim,
    raw::TraitObject,
    sync::{Arc, RwLock},
};

pub fn correct_shading_normal(
    isect: &SurfaceInteraction,
    wo: &Vector3f,
    wi: &Vector3f,
    mode: TransportMode,
) -> Float {
    if let TransportMode::Radiance = mode {
        let num = wo.abs_dot(&isect.shading.n) * wi.abs_dot(&isect.n);
        let denom = wo.abs_dot(&isect.n) * wi.abs_dot(&isect.shading.n);
        if denom == 0.0 {
            0.0
        } else {
            num / denom
        }
    } else {
        1.0
    }
}

#[derive(Deref, DerefMut, Default, Clone)]
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

#[derive(Clone)]
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
pub struct ScopedAssignment<'a, T: Default + Clone> {
    target: Option<&'a mut T>,
    backup: T,
}

impl<'a, T: Default + Clone> Drop for ScopedAssignment<'a, T> {
    fn drop(&mut self) {
        if self.target.is_some() {
            *self.target.take().unwrap() = self.backup.clone();
        }
    }
}

impl<'a, T: Default + Clone> ScopedAssignment<'a, T> {
    pub fn new(mut target: Option<&'a mut T>, value: T) -> Self {
        let mut backup = value.clone();
        if target.is_some() {
            let p_target = target.take().unwrap();
            backup = p_target.clone();
            *p_target = value;
            target = Some(p_target);
        }
        Self { target, backup }
    }

    pub fn replace(&mut self, other: &mut ScopedAssignment<'a, T>) {
        if self.target.is_some() {
            *self.target.take().unwrap() = self.backup.clone();
        }
        self.target = other.target.take();
        self.backup = other.backup.clone();
    }
}

pub struct LightKey(pub LightDt);

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
        pdf += light.pdf_li(&BaseInteraction::default(), &-*w) * light_distr.func[index];
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
    light_sample_strategy: String,
}

impl BDPTIntegrator {
    pub fn new(
        sampler: SamplerDtRw,
        camera: CameraDt,
        max_depth: usize,
        visualize_strategies: bool,
        visualize_weights: bool,
        pixel_bounds: Bounds2i,
        light_sample_strategy: String,
    ) -> Self {
        Self {
            sampler,
            camera,
            max_depth,
            visualize_strategies,
            visualize_weights,
            pixel_bounds,
            light_sample_strategy,
        }
    }
}

impl Integrator for BDPTIntegrator {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn render(&mut self, scene: &Scene) {
        let light_distribution =
            create_light_sample_distribution(self.light_sample_strategy.clone(), scene);
        let mut light2index = HashMap::new();
        for i in 0..scene.lights.len() {
            light2index.insert(LightKey(scene.lights[i].clone()), i);
        }

        let film = self.camera.film();
        let sample_bounds = film.read().unwrap().get_sample_bounds();
        let sample_extent = sample_bounds.diagonal();
        let tile_size = 16;
        let n_x_tiles = (sample_extent.x + tile_size - 1) / tile_size;
        let n_y_tiles = (sample_extent.y + tile_size - 1) / tile_size;

        let buffer_count = (1 + self.max_depth) * (6 + self.max_depth) / 2;
        let weight_films = RwLock::new(HashMap::new());
        if self.visualize_strategies || self.visualize_weights {
            for depth in 0..self.max_depth + 1 {
                for s in 0..depth + 3 {
                    let t = depth + 2 - s;
                    if t == 0 || (s == 1 && t == 1) {
                        continue;
                    }

                    let filename = format!("bdpt_d{}_s{}_t{}.exr", depth, s, t);
                    weight_films.write().unwrap().insert(
                        buffer_index(s, t),
                        Film::new(
                            film.read().unwrap().full_resolution,
                            Bounds2f::from((Point2f::new(0.0, 0.0), Point2f::new(1.0, 1.0))),
                            create_box_filter(),
                            film.read().unwrap().diagonal * 1000.0,
                            filename,
                            1.0,
                            Float::INFINITY,
                        ),
                    );
                }
            }
        }

        if scene.lights.len() > 0 {
            parallel_for_2d!(
                |tile: Point2i| {
                    let seed = tile.y * n_x_tiles + tile.x;
                    let tile_sampler = self.sampler.read().unwrap().clone_sampler(seed as usize);
                    let x0 = sample_bounds.min.x + tile.x * tile_size;
                    let x1 = std::cmp::min(x0 + tile_size, sample_bounds.max.x);
                    let y0 = sample_bounds.min.y + tile.y * tile_size;
                    let y1 = std::cmp::min(y0 + tile_size, sample_bounds.max.y);
                    let tile_bounds = Bounds2i::from((Point2i::new(x0, y0), Point2i::new(x1, y1)));

                    let mut film_tile = self
                        .camera
                        .film()
                        .read()
                        .unwrap()
                        .get_film_tile(&tile_bounds);

                    for pixel in &tile_bounds {
                        tile_sampler.write().unwrap().start_pixel(pixel);
                        if !self.pixel_bounds.inside(&pixel) {
                            continue;
                        }

                        loop {
                            let p_film =
                                tile_sampler.write().unwrap().get_2d() + Point2f::from(pixel);
                            let mut camera_vertices = vec![Vertex::default(); self.max_depth + 2];
                            let mut light_vertices = vec![Vertex::default(); self.max_depth + 1];
                            let n_camera = generate_camera_sub_path(
                                scene,
                                tile_sampler.clone(),
                                self.max_depth + 2,
                                self.camera.clone(),
                                p_film,
                                camera_vertices.as_mut_slice(),
                                0,
                            );

                            let light_distr =
                                light_distribution.lookup(camera_vertices[0].p(), Some(scene));

                            let n_light = generate_light_sub_path(
                                scene,
                                tile_sampler.clone(),
                                self.max_depth + 1,
                                camera_vertices[0].time(),
                                light_distr,
                                &light2index,
                                light_vertices.as_mut_slice(),
                                0,
                            );

                            let mut l = Spectrum::new(0.0);
                            for t in 1..n_camera + 1 {
                                for s in 0..n_light + 1 {
                                    let depth = t + s - 2;
                                    if (s == 1 && t == 1) || depth < 0 || depth > self.max_depth {
                                        continue;
                                    }

                                    let mut p_film_new = p_film;
                                    let mut mis_weight = 0.0;
                                    let l_path = connect_bdpt(
                                        scene,
                                        light_vertices.as_mut_slice(),
                                        camera_vertices.as_mut_slice(),
                                        s,
                                        t,
                                        light_distr,
                                        &light2index,
                                        self.camera.clone(),
                                        tile_sampler.clone(),
                                        &mut p_film_new,
                                        Some(&mut mis_weight),
                                    );

                                    if self.visualize_weights || self.visualize_strategies {
                                        let mut value = Spectrum::default();
                                        if self.visualize_strategies {
                                            value = if mis_weight == 0.0 {
                                                0.0.into()
                                            } else {
                                                l_path / mis_weight
                                            };
                                        }
                                        if self.visualize_weights {
                                            value = l_path;
                                        }
                                        weight_films
                                            .write()
                                            .unwrap()
                                            .get_mut(&buffer_index(s, t))
                                            .unwrap()
                                            .add_splat(&p_film_new, value);
                                    }

                                    if t != 1 {
                                        l += l_path;
                                    } else {
                                        film.write().unwrap().add_splat(&p_film_new, l_path);
                                    }
                                }
                            }
                            film_tile.add_sample(
                                &p_film,
                                l,
                                1.0,
                                &film.read().unwrap().filter_table,
                            );
                            if !tile_sampler.write().unwrap().start_next_sample() {
                                break;
                            }
                        }
                    }
                    film.write().unwrap().merge_film_tile(film_tile);
                },
                Point2i::new(n_x_tiles, n_y_tiles)
            );
        }
    }
}

#[derive(Default, Clone)]
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

    pub fn create_medium(mi: MediumInteraction, beta: Spectrum, pdf: Float, prev: &Vertex) -> Self {
        let mut v = Vertex::from((mi, beta));
        v.pdf_fwd = prev.convert_density(pdf, &v);
        v
    }

    pub fn create_surface(
        si: SurfaceInteraction,
        beta: Spectrum,
        pdf: Float,
        prev: &Vertex,
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

    pub fn get_interaction_dt(&self) -> InteractionDt {
        match self.typ {
            VertexType::Medium => Arc::new(Box::new(self.mi.clone())),
            VertexType::Surface => Arc::new(Box::new(self.si.clone())),
            _ => Arc::new(Box::new(self.ei.clone())),
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
            VertexType::Medium => {
                Spectrum::new(self.mi.phase.as_ref().unwrap().p(&self.mi.wo, &wi))
            }
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
                pdf = self.mi.phase.as_ref().unwrap().p(&wp, &wn);
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

pub fn generate_camera_sub_path(
    scene: &Scene,
    sampler: SamplerDtRw,
    max_depth: usize,
    camera: CameraDt,
    p_film: Point2f,
    path: &mut [Vertex],
    offset: usize,
) -> usize {
    if max_depth == 0 {
        return 0;
    }

    let mut camera_sample = CameraSample::default();
    camera_sample.p_film = p_film;
    camera_sample.time = sampler.write().unwrap().get_1d();
    camera_sample.p_lens = sampler.write().unwrap().get_2d();
    let mut ray = RayDifferentials::default();
    let beta = camera.generate_ray_differential(&camera_sample, &mut ray);
    ray.scale_differentials(1.0 / sampler.read().unwrap().samples_per_pixel() as Float);

    let (mut pdf_pos, mut pdf_dir) = (0.0, 0.0);
    path[offset] = Vertex::create_camera(camera.clone(), &ray.base, beta.into());
    camera.pdf_we(&ray.base, &mut pdf_pos, &mut pdf_dir);
    random_walk(
        scene,
        ray,
        sampler,
        beta.into(),
        pdf_dir,
        max_depth - 1,
        TransportMode::Radiance,
        path,
        offset + 1,
    ) + 1
}

fn random_walk(
    scene: &Scene,
    mut ray: RayDifferentials,
    sampler: SamplerDtRw,
    mut beta: Spectrum,
    pdf: Float,
    max_depth: usize,
    mode: TransportMode,
    path: &mut [Vertex],
    offset: usize,
) -> usize {
    if max_depth == 0 {
        return 0;
    }
    let mut bounces = 0;
    let (mut pdf_fwd, mut pdf_rev) = (pdf, 0.0);
    loop {
        let mut mi = MediumInteraction::default();
        let mut isect = SurfaceInteraction::default();
        let found_interaction = scene.intersect(&mut ray.base, &mut isect);
        if let Some(medium) = &ray.medium {
            medium.sample(&ray.base, sampler.clone(), &mut mi);
        }
        if beta.is_black() {
            break;
        }
        let prev_index = bounces - 1 + offset;
        let prev = path[prev_index].clone(); //TODO fixme
        let vertex = &mut path[offset + bounces];
        if mi.is_valid() {
            *vertex = Vertex::create_medium(mi.clone(), beta, pdf_fwd, &prev);
            bounces += 1;
            if bounces >= max_depth {
                break;
            }

            let mut wi = Vector3f::default();
            pdf_rev = mi.phase.as_ref().unwrap().sample_p(
                &-ray.d,
                &mut wi,
                &sampler.write().unwrap().get_2d(),
            );
            pdf_fwd = pdf_rev;
            ray = mi.spawn_ray(&wi).into();
        } else {
            if !found_interaction {
                if let TransportMode::Radiance = mode {
                    *vertex =
                        Vertex::create_light2(EndPointInteraction::from(&ray.base), beta, pdf_fwd);
                    bounces += 1;
                }
                break;
            }

            isect.compute_scattering_functions(&ray, true, mode);
            if isect.bsdf.is_none() {
                ray = isect.spawn_ray(&ray.d).into();
                continue;
            }

            *vertex = Vertex::create_surface(isect.clone(), beta, pdf_fwd, &prev);
            bounces += 1;
            if bounces >= max_depth {
                break;
            }

            let mut wi = Vector3f::default();
            let wo = &isect.wo;
            let mut typ = BxDFType::empty();
            let f = isect.bsdf.as_ref().unwrap().sample_f(
                wo,
                &mut wi,
                &sampler.write().unwrap().get_2d(),
                &mut pdf_fwd,
                BxDFType::all(),
                Some(&mut typ),
            );

            if f.is_black() || pdf_fwd == 0.0 {
                break;
            }

            beta *= f * wi.abs_dot(&isect.shading.n) / pdf_fwd;

            pdf_rev = isect.bsdf.clone().unwrap().pdf(&wi, wo, BxDFType::all());

            if !(typ & BxDFType::BSDF_SPECULAR).is_empty() {
                vertex.delta = true;
                pdf_rev = 0.0;
                pdf_rev = 0.0;
            }
            beta *= correct_shading_normal(&isect, wo, &wi, mode);
            ray = isect.spawn_ray(&wi).into();
        }

        path[prev_index].pdf_rev = vertex.convert_density(pdf_rev, &prev);
    }
    1
}

pub fn generate_light_sub_path(
    scene: &Scene,
    sampler: SamplerDtRw,
    max_depth: usize,
    time: Float,
    light_distr: &Distribution1D,
    light2index: &HashMap<LightKey, usize>,
    path: &mut [Vertex],
    offset: usize,
) -> usize {
    if max_depth == 0 {
        return 0;
    }

    let mut light_pdf = 0.0;
    let light_num = light_distr.sample_discrete(
        sampler.write().unwrap().get_1d(),
        Some(&mut light_pdf),
        None,
    );
    let light = scene.lights[light_num].clone();
    let mut ray = RayDifferentials::default();
    let mut n_light = Normal3f::default();
    let (mut pdf_pos, mut pdf_dir) = (0.0, 0.0);
    let le = light.sample_le(
        &sampler.write().unwrap().get_2d(),
        &sampler.write().unwrap().get_2d(),
        time,
        &mut ray.base,
        &mut n_light,
        &mut pdf_pos,
        &mut pdf_dir,
    );

    if pdf_pos == 0.0 || pdf_dir == 0.0 || le.is_black() {
        return 0;
    }

    path[offset] = Vertex::create_light(light.clone(), &ray.base, n_light, le, pdf_pos * light_pdf);

    let beta = le * n_light.abs_dot(&ray.d) / (light_pdf * pdf_pos * pdf_dir);
    let d = ray.d;
    let n_vertices = random_walk(
        scene,
        ray,
        sampler,
        beta,
        pdf_dir,
        max_depth - 1,
        TransportMode::Radiance,
        path,
        offset + 1,
    );

    if path[offset].is_infinite_light() {
        if n_vertices > 0 {
            path[offset + 1].pdf_fwd = pdf_pos;
            if path[offset + 1].is_on_surface() {
                path[offset + 1].pdf_fwd *= d.abs_dot(path[offset + 1].ng());
            }
        }
        path[offset].pdf_fwd = infinite_light_density(scene, light_distr, light2index, &d);
    }
    n_vertices + 1
}

pub fn connect_bdpt(
    scene: &Scene,
    light_vertices: &mut [Vertex],
    camera_vertices: &mut [Vertex],
    s: usize,
    t: usize,
    light_distr: &Distribution1D,
    light2index: &HashMap<LightKey, usize>,
    camera: CameraDt,
    sampler: SamplerDtRw,
    p_raster: &mut Point2f,
    mis_weight_ptr: Option<&mut Float>,
) -> Spectrum {
    let mut l = Spectrum::new(0.0);

    if t > 1 && s != 0 {
        if let VertexType::Light = camera_vertices[t - 1].typ {
            return l;
        }
    }

    let mut sampled = Vertex::default();
    if s == 0 {
        let pt = &camera_vertices[t - 1];
        if pt.is_light() {
            l = pt.le(scene, &camera_vertices[t - 2]) * pt.beta;
        }
    } else if t == 1 {
        let qs = &light_vertices[s - 1];
        if qs.is_connectible() {
            let mut vis = VisibilityTester::default();
            let mut wi = Vector3f::default();
            let mut pdf = 0.0;
            let swi = camera.sample_wi(
                qs.get_interaction_dt(),
                &sampler.write().unwrap().get_2d(),
                &mut wi,
                &mut pdf,
                Some(p_raster),
                &mut vis,
            );
            if pdf > 0.0 && swi.is_black() {
                sampled =
                    Vertex::create_camera2(camera.clone(), vis.p1().as_base().clone(), swi / pdf);
                l = qs.beta * qs.f(&sampled, TransportMode::Importance) * sampled.beta;
                if qs.is_on_surface() {
                    l *= wi.abs_dot(qs.ns());
                }
                if l.is_black() {
                    l *= vis.tr(scene, sampler.clone());
                }
            }
        }
    } else if s == 1 {
        let pt = &camera_vertices[t - 1];
        if pt.is_connectible() {
            let mut light_pdf = 0.0;
            let mut vis = VisibilityTester::default();
            let mut wi = Vector3f::default();
            let mut pdf = 0.0;
            let light_num = light_distr.sample_discrete(
                sampler.write().unwrap().get_1d(),
                Some(&mut light_pdf),
                None,
            );

            let light = scene.lights[light_num].clone();
            let light_weight = light.sample_li(
                pt.get_interaction(),
                &sampler.write().unwrap().get_2d(),
                &mut wi,
                &mut pdf,
                &mut vis,
            );

            if pdf > 0.0 && !light_weight.is_black() {
                let ei = EndPointInteraction::from((vis.p1().as_base().clone(), light.clone()));
                sampled = Vertex::create_light2(ei, light_weight / (pdf * light_pdf), 0.0);
                sampled.pdf_fwd = sampled.pdf_light_origin(scene, pt, light_distr, light2index);
                l = pt.beta * pt.f(&sampled, TransportMode::Importance) * sampled.beta;
                if pt.is_on_surface() {
                    l *= wi.abs_dot(pt.ns());
                }
                if !l.is_black() {
                    l *= vis.tr(scene, sampler.clone());
                }
            }
        }
    } else {
        let qs = &light_vertices[s - 1];
        let pt = &camera_vertices[t - 1];

        if qs.is_connectible() && pt.is_connectible() {
            l = qs.beta
                * qs.f(pt, TransportMode::Importance)
                * pt.f(qs, TransportMode::Radiance)
                * pt.beta;
            if !l.is_black() {
                l *= g(scene, sampler, qs, pt);
            }
        }
    }
    let mis_weight = if l.is_black() {
        0.0
    } else {
        mis_weight(
            scene,
            light_vertices,
            camera_vertices,
            sampled,
            s,
            t,
            light_distr,
            light2index,
        )
    };
    l *= mis_weight;
    if let Some(mis_weight_ptr) = mis_weight_ptr {
        *mis_weight_ptr = mis_weight;
    }
    l
}

fn g(scene: &Scene, sampler: SamplerDtRw, v0: &Vertex, v1: &Vertex) -> Spectrum {
    let mut d = *v0.p() - *v1.p();
    let mut g = 1.0 / d.length_squared();
    d *= g.sqrt();
    if v0.is_on_surface() {
        g *= v0.ns().abs_dot(&d);
    }
    if v1.is_on_surface() {
        g *= v1.ns().abs_dot(&d);
    }

    let vis = VisibilityTester::new(v0.get_interaction_dt(), v1.get_interaction_dt());
    vis.tr(scene, sampler) * g
}

fn mis_weight(
    scene: &Scene,
    light_vertices: &[Vertex],
    camera_vertices: &[Vertex],
    sampled: Vertex,
    s: usize,
    t: usize,
    light_pdf: &Distribution1D,
    light2index: &HashMap<LightKey, usize>,
) -> Float {
    if s + t == 2 {
        return 0.0;
    }
    let mut sum_ri = 0.0;
    let remap = |f: Float| {
        if f != 0.0 {
            f
        } else {
            1.0
        }
    };

    let mut ri = 1.0;
    for i in (1..t).rev() {
        ri *= remap(camera_vertices[i].pdf_rev) / remap(camera_vertices[i].pdf_fwd);
        if !camera_vertices[i].delta && !camera_vertices[i - 1].delta {
            sum_ri += ri;
        }
    }

    ri = 1.0;
    for i in (0..s).rev() {
        ri *= remap(light_vertices[i].pdf_rev) / remap(light_vertices[i].pdf_fwd);
        let delta_light_vertex = if i > 0 {
            light_vertices[i - 1].delta
        } else {
            light_vertices[0].is_delta_light()
        };
        if !light_vertices[i].delta && !delta_light_vertex {
            sum_ri += ri;
        }
    }
    1.0 / (1.0 + sum_ri)
}

#[inline]
fn buffer_index(s: usize, t: usize) -> usize {
    let above = s + t - 2;
    s + above * (5 + above) / 2
}
