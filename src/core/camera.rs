use crate::core::film::Film;
use crate::core::geometry::{Point2f, Ray, RayDifferentials, Vector3f};
use crate::core::interaction::Interaction;
use crate::core::light::VisibilityTester;
use crate::core::medium::Medium;
use crate::core::spectrum::Spectrum;
use crate::core::transform::{AnimatedTransform, Transformf};
use crate::Float;
use std::any::Any;
use std::sync::Arc;

#[derive(Default, Copy, Clone)]
pub struct CameraSample {
    p_film: Point2f,
    p_lens: Point2f,
    time: Float,
}
pub trait Camera {
    fn as_any(&self) -> &dyn Any;
    fn generate_ray(&self, sample: &CameraSample, ray: &mut Ray) -> Float;
    fn generate_ray_differential(&self, sample: &CameraSample, rd: &mut RayDifferentials) -> Float {
        let wt = self.generate_ray(sample, rd);
        if wt == 0.0 {
            return wt;
        }
        let mut wtx = 0.0;
        for eps in &[0.05, -0.05] {
            let mut sshift = *sample;
            sshift.p_film.x += *eps;
            let mut rx = Ray::default();
            wtx = self.generate_ray(&sshift, &mut rx);
            rd.rx_origin = rd.o + (rx.o - rd.o) / *eps;
            rd.rx_direction = rd.d + (rx.d - rd.d) / *eps;
            if wtx != 0.0 {
                break;
            }
        }
        if wtx == 0.0 {
            return wtx;
        }
        let mut wty = 0.0;
        for eps in &[0.05, -0.05] {
            let mut sshift = *sample;
            sshift.p_film.y += *eps;
            let mut ry = Ray::default();
            wty = self.generate_ray(&sshift, &mut ry);
            rd.ry_origin = rd.o + (ry.o - rd.o) / *eps;
            rd.ry_direction = rd.d + (ry.d - rd.d) / *eps;
            if wty == 0.0 {
                break;
            }
        }
        if wty == 0.0 {
            return wty;
        }
        rd.has_differentials = true;
        wt
    }
    fn we(&self, ray: &Ray, p_raster2: Option<&mut Point2f>) -> Spectrum {
        unimplemented!()
    }
    fn pdf_we(&self, ray: &Ray, pdf_pos: &mut Float, pdf_dir: &mut Float) {
        unimplemented!()
    }
    fn sample_wi(
        &self,
        refer: &Interaction,
        sample: &Point2f,
        wi: &mut Vector3f,
        pdf: &mut Float,
        p_raster: Option<&mut Point2f>,
        vis: &mut VisibilityTester,
    ) -> Spectrum {
        unimplemented!()
    }
    fn camera_to_world(&self) -> &AnimatedTransform;
    fn shutter_open(&self) -> Float;
    fn shutter_close(&self) -> Float;
    fn film(&self) -> Arc<Film>;
    fn medium(&self) -> Arc<Box<dyn Medium>>;
}

pub(crate) struct BaseCamera {
    pub camera_to_world: AnimatedTransform,
    pub shutter_open: Float,
    pub shutter_close: Float,
    pub film: Arc<Film>,
    pub medium: Arc<Box<dyn Medium>>,
}

impl BaseCamera {
    pub fn new(
        camera_to_world: AnimatedTransform,
        shutter_open: Float,
        shutter_close: Float,
        film: Arc<Film>,
        medium: Arc<Box<dyn Medium>>,
    ) -> BaseCamera {
        if camera_to_world.has_scale() {
            log::warn!(
                "scaling detected in world-to-camera transformation!\n\
             The system has numerous assumptions, implicit and explicit,\n\
             that this transform will have no scale factors in it.\n\
             Proceed at your own risk; your image may have errors or \n\
             the system may crash as a result of this."
            )
        }
        Self {
            camera_to_world,
            shutter_open,
            shutter_close,
            film,
            medium,
        }
    }
}

#[macro_export]
macro_rules! impl_base_camera {
    () => {
        fn camera_to_world(&self) -> &crate::core::transform::AnimatedTransform {
            &self.base.camera_to_world
        }

        fn shutter_open(&self) -> crate::Float {
            self.base.shutter_open
        }

        fn shutter_close(&self) -> crate::Float {
            self.base.shutter_close
        }

        fn film(&self) -> std::sync::Arc<crate::core::film::Film> {
            self.base.film.clone()
        }

        fn medium(&self) -> std::sync::Arc<Box<dyn crate::core::medium::Medium>> {
            self.base.medium.clone()
        }
    };
}
