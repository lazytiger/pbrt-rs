use crate::{
    core::{
        geometry::{Bounds2f, Normal3f, Point2f, Point2i, Point3f, Ray, Vector3f},
        imageio::read_image,
        interaction::{BaseInteraction, Interaction},
        light::{BaseLight, Light, LightFlags, VisibilityTester},
        medium::MediumInterface,
        mipmap::{ImageWrap, MIPMap},
        pbrt::{Float, PI},
        reflection::cos_theta,
        sampling::{uniform_cone_pdf, uniform_sample_clone},
        spectrum::{RGBSpectrum, Spectrum, SpectrumType},
        transform::{Point3Ref, Transformf, Vector3Ref},
    },
    impl_base_light,
};
use derive_more::{Deref, DerefMut};
use std::{any::Any, sync::Arc};

#[derive(Debug, Deref, DerefMut)]
pub struct ProjectionLight {
    #[deref]
    #[deref_mut]
    base: BaseLight,
    projection_map: Option<MIPMap<RGBSpectrum>>,
    p_light: Point3f,
    i: Spectrum,
    light_projection: Transformf,
    hither: Float,
    yon: Float,
    screen_bounds: Bounds2f,
    cos_total_width: Float,
}

impl ProjectionLight {
    pub fn new(
        light_to_world: Transformf,
        medium: MediumInterface,
        i: Spectrum,
        texname: String,
        fov: Float,
    ) -> Self {
        let p_light = &light_to_world * Point3Ref(&Point3f::default());
        let base = BaseLight::new(LightFlags::DELTA_POSITION, light_to_world, medium, 1);
        let mut resolution = Point2i::default();
        let texels = read_image(texname, &mut resolution);
        let mut projection_map = None;
        if let Some(texels) = texels {
            projection_map.replace(MIPMap::new(
                resolution,
                texels,
                false,
                8.0,
                ImageWrap::Repeat,
            ));
        }
        let aspect = if projection_map.is_some() {
            resolution.x as Float / resolution.y as Float
        } else {
            1.0
        };
        let screen_bounds = if aspect > 1.0 {
            Bounds2f::from((Point2f::new(-aspect, -1.0), Point2f::new(aspect, 1.0)))
        } else {
            Bounds2f::from((
                Point2f::new(-1.0, -1.0 / aspect),
                Point2f::new(1.0, 1.0 / aspect),
            ))
        };
        let hither = 1e-3;
        let yon = 1e30;
        let light_projection = Transformf::perspective(fov, hither, yon);
        let screen_to_light = light_projection.inverse();
        let p_corner = Point3f::new(screen_bounds.max.x, screen_bounds.max.y, 0.0);
        let w_corner = (&screen_to_light * Point3Ref(&p_corner)).normalize();
        let cos_total_width = w_corner.z;
        Self {
            base,
            projection_map,
            p_light,
            i,
            light_projection,
            hither,
            yon,
            screen_bounds,
            cos_total_width,
        }
    }

    pub fn projection(&self, w: &Vector3f) -> Spectrum {
        let wl = &self.world_to_light * Vector3Ref(w);
        if wl.z < self.hither {
            return 0.0.into();
        }
        let p = &self.light_projection * Point3Ref(&wl);
        let p = Point2f::new(p.x, p.y);
        if !self.screen_bounds.inside(&p) {
            return 0.0.into();
        }
        if self.projection_map.is_none() {
            return 1.0.into();
        }
        let st = self.screen_bounds.offset(&p);
        self.projection_map.as_ref().unwrap().lookup(&st, 0.0)
    }
}

impl Light for ProjectionLight {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn sample_li(
        &self,
        iref: &dyn Interaction,
        u: &Point2f,
        wi: &mut Vector3f,
        pdf: &mut f32,
        vis: &mut VisibilityTester,
    ) -> Spectrum {
        *wi = (self.p_light - iref.as_base().p).normalize();
        *pdf = 1.0;
        *vis = VisibilityTester::new(
            Arc::new(Box::new(iref.as_base().clone())),
            Arc::new(Box::new(BaseInteraction::from((
                self.p_light,
                iref.as_base().time,
                self.medium_interface.clone(),
            )))),
        );
        self.i / self.projection(&-*wi) / self.p_light.distance_square(&iref.as_base().p)
    }

    fn power(&self) -> Spectrum {
        if let Some(projection_map) = &self.projection_map {
            projection_map.lookup(&Point2f::new(0.5, 0.5), 0.5)
        } else {
            Spectrum::new(1.0) * self.i * (2.0 * PI * (1.0 - self.cos_total_width))
        }
    }

    fn pdf_li(&self, iref: &dyn Interaction, wi: &Vector3f) -> f32 {
        0.0
    }

    fn sample_le(
        &self,
        u1: &Point2f,
        u2: &Point2f,
        time: f32,
        ray: &mut Ray,
        n_light: &mut Normal3f,
        pdf_pos: &mut f32,
        pdf_dir: &mut f32,
    ) -> Spectrum {
        let v = uniform_sample_clone(u1, self.cos_total_width);
        *ray = Ray::new(
            self.p_light,
            &self.light_to_world * Vector3Ref(&v),
            Float::INFINITY,
            time,
            self.medium_interface.inside.clone(),
        );
        *n_light = ray.d;
        *pdf_pos = 1.0;
        *pdf_dir = uniform_cone_pdf(self.cos_total_width);
        self.i * self.projection(&ray.d)
    }

    fn pdf_le(&self, ray: &Ray, n_light: &Vector3f, pdf_pos: &mut f32, pdf_dir: &mut f32) {
        *pdf_pos = 0.0;
        *pdf_dir = if cos_theta(&(&self.world_to_light * Vector3Ref(&ray.d))) > self.cos_total_width
        {
            uniform_cone_pdf(self.cos_total_width)
        } else {
            0.0
        };
    }

    impl_base_light!();
}
