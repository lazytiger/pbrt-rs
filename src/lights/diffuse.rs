use crate::{
    core::{
        geometry::{Normal3f, Point2f, Ray, Vector3f},
        interaction::{BaseInteraction, Interaction, InteractionDt, SurfaceInteraction},
        light::{BaseLight, Light, LightFlags, VisibilityTester},
        medium::MediumInterface,
        pbrt::{Float, ONE_MINUS_EPSILON, PI},
        sampling::{cosine_hemisphere_pdf, cosine_sample_hemisphere},
        shape::ShapeDt,
        spectrum::Spectrum,
        transform::Transformf,
    },
    impl_base_light,
};
use derive_more::{Deref, DerefMut};
use std::{any::Any, sync::Arc};

#[derive(Debug, Deref, DerefMut)]
pub struct DiffuseAreaLight {
    #[deref]
    #[deref_mut]
    base: BaseLight,
    l_emit: Spectrum,
    shape: ShapeDt,
    two_sided: bool,
    area: Float,
}

impl DiffuseAreaLight {
    pub fn new(
        light_to_world: Transformf,
        medium_interface: MediumInterface,
        le: Spectrum,
        n_samples: usize,
        shape: ShapeDt,
        two_sided: bool,
    ) -> Self {
        let base = BaseLight::new(
            LightFlags::AREA,
            light_to_world,
            medium_interface,
            n_samples,
        );
        let area = shape.area();
        Self {
            base,
            area,
            l_emit: le,
            two_sided,
            shape,
        }
    }
}

impl Light for DiffuseAreaLight {
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
        let ref_dt: InteractionDt = Arc::new(Box::new(iref.as_base().clone()));
        let mut p_shape = self.shape.sample2(ref_dt.clone(), u, pdf);
        Arc::get_mut(&mut p_shape)
            .unwrap()
            .as_base_mut()
            .medium_interface = self.medium_interface.clone();
        if *pdf == 0.0 || (p_shape.as_base().p - iref.as_base().p).length_squared() == 0.0 {
            *pdf = 0.0;
            return 0.0.into();
        }
        *wi = (p_shape.as_base().p - iref.as_base().p).normalize();
        *vis = VisibilityTester::new(ref_dt, p_shape.clone());
        self.l(p_shape.as_any().downcast_ref().unwrap(), &-*wi)
    }

    fn power(&self) -> Spectrum {
        self.l_emit * (if self.two_sided { 2.0 } else { 1.0 } * self.area * PI)
    }

    fn pdf_li(&self, iref: &dyn Interaction, wi: &Vector3f) -> f32 {
        let ref_dt: InteractionDt = Arc::new(Box::new(iref.as_base().clone()));
        self.shape.pdf2(ref_dt, wi)
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
        let mut p_shape = self.shape.sample(u1, pdf_pos);
        Arc::get_mut(&mut p_shape)
            .unwrap()
            .as_base_mut()
            .medium_interface = self.medium_interface.clone();
        *n_light = p_shape.as_base().n;
        let mut w = Vector3f::default();
        if self.two_sided {
            let mut u = *u2;
            if u[0] < 0.5 {
                u[0] = ONE_MINUS_EPSILON.min(u[0] * 2.0);
                w = cosine_sample_hemisphere(&u);
            } else {
                u[0] = ONE_MINUS_EPSILON.min((u[0] - 0.5) * 2.0);
                w = cosine_sample_hemisphere(&u);
                w.z *= -1.0;
            }

            *pdf_dir = 0.5 * cosine_hemisphere_pdf(w.z.abs());
        } else {
            w = cosine_sample_hemisphere(u2);
            *pdf_dir = cosine_hemisphere_pdf(w.z);
        }

        let n = p_shape.as_base().n;
        let (v1, v2) = n.coordinate_system();
        w = v1 * w.x + v2 * w.y + n * w.z;
        *ray = p_shape.as_base().spawn_ray(&w);
        self.l(p_shape.as_any().downcast_ref().unwrap(), &w)
    }

    fn pdf_le(&self, ray: &Ray, n_light: &Vector3f, pdf_pos: &mut f32, pdf_dir: &mut f32) {
        let it = BaseInteraction::new(
            ray.o,
            *n_light,
            Vector3f::default(),
            *n_light,
            ray.time,
            self.medium_interface.clone(),
        );
        *pdf_pos = self.shape.pdf(Arc::new(Box::new(it)));
        *pdf_dir = if self.two_sided {
            0.5 * cosine_hemisphere_pdf(n_light.abs_dot(&ray.d))
        } else {
            cosine_hemisphere_pdf(n_light.dot(&ray.d))
        };
    }

    fn l(&self, si: &SurfaceInteraction, v: &Vector3f) -> Spectrum {
        if self.two_sided || si.n.dot(v) > 0.0 {
            self.l_emit
        } else {
            0.0.into()
        }
    }

    impl_base_light!();
}
