use crate::{
    core::{
        geometry::{Normal3f, Point2f, Point3f, Ray, Vector3f},
        interaction::{BaseInteraction, Interaction},
        light::{BaseLight, Light, LightFlags, VisibilityTester},
        medium::MediumInterface,
        pbrt::{radians, Float, PI},
        reflection::cos_theta,
        sampling::{uniform_cone_pdf, uniform_sample_clone},
        spectrum::Spectrum,
        transform::{Point3Ref, Transformf, Vector3Ref},
    },
    impl_base_light,
};
use derive_more::{Deref, DerefMut};
use std::{any::Any, sync::Arc};

#[derive(Debug, Deref, DerefMut)]
pub struct SpotLight {
    #[deref]
    #[deref_mut]
    base: BaseLight,
    p_light: Point3f,
    i: Spectrum,
    cos_total_width: Float,
    cos_falloff_start: Float,
}

impl SpotLight {
    pub fn new(
        light_to_world: Transformf,
        m: MediumInterface,
        i: Spectrum,
        total_width: Float,
        falloff_start: Float,
    ) -> Self {
        let p_light = &light_to_world * Point3Ref(&Point3f::default());
        let cos_total_width = radians(total_width).cos();
        let cos_falloff_start = radians(falloff_start).cos();
        let base = BaseLight::new(LightFlags::DELTA_POSITION, light_to_world, m, 1);
        Self {
            base,
            p_light,
            cos_falloff_start,
            cos_total_width,
            i,
        }
    }

    pub fn falloff(&self, w: &Vector3f) -> Float {
        let wl = &self.world_to_light * Vector3Ref(w);
        let cos_theta = wl.z;
        if cos_theta < self.cos_total_width {
            0.0
        } else if cos_theta >= self.cos_falloff_start {
            1.0
        } else {
            let delta = (cos_theta - self.cos_total_width)
                / (self.cos_falloff_start - self.cos_total_width);
            (delta * delta) * (delta * delta)
        }
    }
}

impl Light for SpotLight {
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
        self.i * self.falloff(&-*wi) / self.p_light.distance_square(&iref.as_base().p)
    }

    fn power(&self) -> Spectrum {
        self.i * (2.0 * PI * (1.0 - 0.5 * (self.cos_falloff_start + self.cos_total_width)))
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
        let w = uniform_sample_clone(u1, self.cos_total_width);
        *ray = Ray::new(
            self.p_light,
            &self.light_to_world * Vector3Ref(&w),
            Float::INFINITY,
            time,
            self.medium_interface.inside.clone(),
        );
        *n_light = ray.d;
        *pdf_pos = 1.0;
        *pdf_dir = uniform_cone_pdf(self.cos_total_width);
        self.i * self.falloff(&ray.d)
    }

    fn pdf_le(&self, ray: &Ray, n_light: &Vector3f, pdf_pos: &mut f32, pdf_dir: &mut f32) {
        *pdf_pos = 0.0;
        *pdf_dir =
            if cos_theta(&(&self.world_to_light * Vector3Ref(&ray.d))) >= self.cos_total_width {
                uniform_cone_pdf(self.cos_total_width)
            } else {
                0.0
            }
    }

    impl_base_light!();
}
