use crate::core::camera::{BaseCamera, Camera, CameraSample};
use crate::core::geometry::{Point2f, Ray, RayDifferentials, Vector3f};
use crate::core::interaction::Interaction;
use crate::core::light::VisibilityTester;
use crate::core::spectrum::Spectrum;
use crate::core::transform::Transformf;
use crate::{impl_base_camera, Float};
use std::any::Any;

pub struct PerspectiveCamera {
    base: BaseCamera,
    dx_camera: Vector3f,
    dy_camera: Vector3f,
    camera_to_screen: Transformf,
    raster_to_camera: Transformf,
    screen_to_raster: Transformf,
    raster_to_screen: Transformf,
    lens_radius: Float,
    focal_distance: Float,
    a: Float,
}

impl Camera for PerspectiveCamera {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn generate_ray(&self, sample: &CameraSample, ray: &mut Ray) -> f32 {
        unimplemented!()
    }

    fn generate_ray_differential(&self, sample: &CameraSample, rd: &mut RayDifferentials) -> f32 {
        unimplemented!()
    }

    fn we(&self, ray: &Ray, p_raster2: Option<&mut Point2f>) -> Spectrum {
        unimplemented!()
    }

    fn pdf_we(&self, ray: &Ray, pdf_pos: &mut f32, pdf_dir: &mut f32) {
        unimplemented!()
    }

    fn sample_wi(
        &self,
        refer: &Interaction,
        sample: &Point2f,
        wi: &mut Vector3f,
        pdf: &mut f32,
        p_raster: Option<&mut Point2f>,
        vis: &mut VisibilityTester,
    ) -> Spectrum {
        unimplemented!()
    }

    impl_base_camera!();
}
