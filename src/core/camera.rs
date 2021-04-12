use crate::core::{
    film::Film,
    geometry::{Point2f, Ray, RayDifferentials, Vector3f},
    interaction::Interaction,
    light::VisibilityTester,
    medium::{Medium, MediumDt},
    pbrt::Float,
    spectrum::Spectrum,
    transform::AnimatedTransform,
};
use std::{
    any::Any,
    sync::{Arc, Mutex, RwLock},
};

#[derive(Default, Copy, Clone)]
pub struct CameraSample {
    pub p_film: Point2f,
    pub p_lens: Point2f,
    pub time: Float,
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
    fn we(&self, _ray: &Ray, _p_raster2: Option<&mut Point2f>) -> Spectrum {
        unimplemented!()
    }
    fn pdf_we(&self, _ray: &Ray, _pdf_pos: &mut Float, _pdf_dir: &mut Float) {
        unimplemented!()
    }
    fn sample_wi(
        &self,
        _refer: &Interaction,
        _sample: &Point2f,
        _wi: &mut Vector3f,
        _pdf: &mut Float,
        _p_raster: Option<&mut Point2f>,
        _vis: &mut VisibilityTester,
    ) -> Spectrum {
        unimplemented!()
    }
    fn camera_to_world(&self) -> &AnimatedTransform;
    fn shutter_open(&self) -> Float;
    fn shutter_close(&self) -> Float;
    fn film(&self) -> Arc<Film>;
    fn medium(&self) -> MediumDt;
}

pub(crate) struct BaseCamera {
    pub camera_to_world: AnimatedTransform,
    pub shutter_open: Float,
    pub shutter_close: Float,
    pub film: Arc<Film>,
    pub medium: MediumDt,
}

impl BaseCamera {
    pub fn new(
        camera_to_world: AnimatedTransform,
        shutter_open: Float,
        shutter_close: Float,
        film: Arc<Film>,
        medium: MediumDt,
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
        #[inline]
        fn camera_to_world(&self) -> &crate::core::transform::AnimatedTransform {
            &self.base.camera_to_world
        }

        #[inline]
        fn shutter_open(&self) -> crate::core::pbrt::Float {
            self.base.shutter_open
        }

        #[inline]
        fn shutter_close(&self) -> crate::core::pbrt::Float {
            self.base.shutter_close
        }

        #[inline]
        fn film(&self) -> std::sync::Arc<crate::core::film::Film> {
            self.base.film.clone()
        }

        #[inline]
        fn medium(&self) -> crate::core::medium::MediumDt {
            self.base.medium.clone()
        }
    };
}

pub type CameraDt = Arc<Box<dyn Camera>>;
pub type CameraDtMut = Arc<Mutex<Box<dyn Camera>>>;
pub type CameraDtRw = Arc<RwLock<Box<dyn Camera>>>;
