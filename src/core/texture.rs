use crate::core::{
    geometry::{spherical_phi, spherical_theta, Point2, Point2f, Point3f, Vector2f, Vector3f},
    interaction::SurfaceInteraction,
    pbrt::{clamp, lerp, Float, INV_2_PI, INV_PI, PI},
    transform::{Point3Ref, Transformf, Vector3Ref},
};
use std::{any::Any, fmt::Debug, sync::Arc};

pub trait TextureMapping2D: Debug {
    fn as_any(&self) -> &dyn Any;
    fn map(&self, si: &SurfaceInteraction, dstdx: &mut Vector2f, dstdy: &mut Vector2f) -> Point2f;
}

pub type TextureMapping2DDt = Arc<Box<dyn TextureMapping2D + Sync + Send>>;

#[derive(Debug)]
pub struct UVMapping2D {
    su: Float,
    sv: Float,
    du: Float,
    dv: Float,
}

impl UVMapping2D {
    pub fn new(su: Float, sv: Float, du: Float, dv: Float) -> Self {
        Self { su, sv, du, dv }
    }
}

impl Default for UVMapping2D {
    fn default() -> Self {
        Self::new(1.0, 1.0, 0.0, 0.0)
    }
}

impl TextureMapping2D for UVMapping2D {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn map(&self, si: &SurfaceInteraction, dstdx: &mut Vector2f, dstdy: &mut Vector2f) -> Point2f {
        *dstdx = Vector2f::new(self.su * si.dudx, self.sv * si.dvdx);
        *dstdy = Vector2f::new(self.su * si.dudy, self.sv * si.dvdy);
        Point2f::new(self.su * si.uv[0] + self.du, self.sv * si.uv[1] + self.dv)
    }
}

#[derive(Default, Debug)]
pub struct SphericalMapping2D {
    world_to_texture: Transformf,
}

impl SphericalMapping2D {
    pub fn new(world_to_texture: Transformf) -> Self {
        Self { world_to_texture }
    }

    fn sphere(&self, p: &Point3f) -> Point2f {
        let vec = (&self.world_to_texture * Point3Ref(p)).normalize() - Point3f::default();
        let theta = spherical_theta(&vec);
        let phi = spherical_phi(&vec);
        Point2f::new(theta * INV_PI, phi * INV_2_PI)
    }
}

impl TextureMapping2D for SphericalMapping2D {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn map(&self, si: &SurfaceInteraction, dstdx: &mut Vector2f, dstdy: &mut Vector2f) -> Point2f {
        let st = self.sphere(&si.p);
        let delta = 0.1;
        let st_delta_x = self.sphere(&(si.p + si.dpdx * delta));
        *dstdx = (st_delta_x - st) / delta;
        let st_delta_y = self.sphere(&(si.p + si.dpdy * delta));
        *dstdy = (st_delta_y - st) / delta;

        if dstdx[1] > 0.5 {
            dstdx[1] = 1.0 - dstdx[1];
        } else if dstdx[1] < -0.5 {
            dstdx[1] = -(dstdx[1] + 1.0);
        }

        if dstdy[1] > 0.5 {
            dstdy[1] = 1.0 - dstdy[1];
        } else if dstdy[1] < -0.5 {
            dstdy[1] = -(dstdy[1] + 1.0);
        }

        st
    }
}

#[derive(Debug)]
pub struct CylindricalMapping2D {
    world_to_texture: Transformf,
}

impl CylindricalMapping2D {
    pub fn new(world_to_texture: Transformf) -> CylindricalMapping2D {
        Self { world_to_texture }
    }

    fn cylinder(&self, p: &Point3f) -> Point2f {
        todo!()
    }
}

impl TextureMapping2D for CylindricalMapping2D {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn map(&self, si: &SurfaceInteraction, dstdx: &mut Vector2f, dstdy: &mut Vector2f) -> Point2f {
        let st = self.cylinder(&si.p);
        let delta = 0.01;
        let st_delta_x = self.cylinder(&(si.p + si.dpdx * delta));
        *dstdx = (st_delta_x - st) / delta;
        if dstdx[1] > 0.5 {
            dstdx[1] = 1.0 - dstdx[1];
        } else if dstdx[1] < 0.5 {
            dstdx[1] = -(dstdx[1] + 1.0);
        }
        let st_delta_y = self.cylinder(&(si.p + si.dpdy * delta));
        *dstdy = (st_delta_y - st) / delta;
        if dstdy[1] > 0.5 {
            dstdy[1] = 1.0 - dstdy[1];
        } else if dstdy[1] < -0.5 {
            dstdy[1] = -(dstdy[1] + 1.0)
        }
        st
    }
}

#[derive(Debug)]
pub struct PlanarMapping2D {
    vs: Vector3f,
    vt: Vector3f,
    ds: Float,
    dt: Float,
}

impl PlanarMapping2D {
    pub fn new(vs: Vector3f, vt: Vector3f, ds: Float, dt: Float) -> Self {
        Self { vs, vt, ds, dt }
    }
}

impl TextureMapping2D for PlanarMapping2D {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn map(&self, si: &SurfaceInteraction, dstdx: &mut Vector2f, dstdy: &mut Vector2f) -> Point2f {
        let vec = &si.p;
        *dstdx = Vector2f::new(si.dpdx.dot(&self.vs), si.dpdx.dot(&self.vt));
        *dstdy = Vector2f::new(si.dpdy.dot(&self.vs), si.dpdy.dot(&self.vt));
        Point2f::new(self.ds + vec.dot(&self.vs), self.dt + vec.dot(&self.vt))
    }
}

pub trait TextureMapping3D: Debug {
    fn as_any(&self) -> &dyn Any;
    fn map(&self, si: &SurfaceInteraction, dpdx: &mut Vector3f, dpdy: &mut Vector3f) -> Point3f;
}

pub type TextureMapping3DDt = Arc<Box<dyn TextureMapping3D + Sync + Send>>;

#[derive(Debug)]
pub struct IdentityMapping3D {
    world_to_texture: Transformf,
}

impl IdentityMapping3D {
    pub fn new(world_to_texture: Transformf) -> Self {
        Self { world_to_texture }
    }
}

impl TextureMapping3D for IdentityMapping3D {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn map(&self, si: &SurfaceInteraction, dpdx: &mut Vector3f, dpdy: &mut Vector3f) -> Point3f {
        *dpdx = &self.world_to_texture * Vector3Ref(&si.dpdx);
        *dpdy = &self.world_to_texture * Vector3Ref(&si.dpdy);
        &self.world_to_texture * Point3Ref(&si.p)
    }
}

pub trait Texture<T>: Debug {
    fn as_any(&self) -> &dyn Any;
    fn evaluate(&self, si: &SurfaceInteraction) -> T;
}

pub type TextureDt<T> = Arc<Box<dyn Texture<T>>>;

pub fn lanczos(x: Float, tau: Float) -> Float {
    let mut x = x.abs();
    if x < 1e-5 {
        return 1.0;
    }
    if x > 1.0 {
        return 0.0;
    }
    x *= PI;
    let s = (x * tau).sin() / (x * tau);
    let lanczos = x.sin() / x;
    s * lanczos
}

pub fn noise(x: Float, y: Float, z: Float) -> Float {
    let mut ix = x.floor() as i32;
    let mut iy = y.floor() as i32;
    let mut iz = z.floor() as i32;
    let dx = x - ix as Float;
    let dy = y - iy as Float;
    let dz = z - iz as Float;

    ix &= NOISE_PERM_SIZE as i32 - 1;
    iy &= NOISE_PERM_SIZE as i32 - 1;
    iz &= NOISE_PERM_SIZE as i32 - 1;

    let w000 = grad(ix, iy, iz, dx, dy, dz);
    let w100 = grad(ix + 1, iy, iz, dx - 1.0, dy, dz);
    let w010 = grad(ix, iy + 1, iz, dx, dy - 1.0, dz);
    let w110 = grad(ix + 1, iy + 1, iz, dx - 1.0, dy - 1.0, dz);
    let w001 = grad(ix, iy, iz + 1, dx, dy, dz - 1.0);
    let w101 = grad(ix + 1, iy, iz + 1, dx - 1.0, dy, dz - 1.0);
    let w011 = grad(ix, iy + 1, iz + 1, dx, dy - 1.0, dz - 1.0);
    let w111 = grad(ix + 1, iy + 1, iz + 1, dx - 1.0, dy - 1.0, dz - 1.0);

    let wx = noise_weight(dx);
    let wy = noise_weight(dy);
    let wz = noise_weight(dz);

    let x00 = lerp(wx, w000, w100);
    let x10 = lerp(wx, w010, w110);
    let x01 = lerp(wx, w001, w101);
    let x11 = lerp(wx, w011, w111);

    let y0 = lerp(wy, x00, x10);
    let y1 = lerp(wy, x01, x11);

    lerp(wz, y0, y1)
}

pub fn noise_point(p: &Point3f) -> Float {
    noise(p.x, p.y, p.z)
}

pub fn fbm(
    p: &Point3f,
    dpdx: &Vector3f,
    dpdy: &Vector3f,
    omega: Float,
    max_octaves: Float,
) -> Float {
    let len2 = dpdx.length_squared().max(dpdy.length_squared());
    let n = clamp(-1.0 - 0.5 * len2.log2(), 0.0, max_octaves);
    let n_int = n.floor() as i32;

    let mut sum = 0.0;
    let mut lambda = 1.0;
    let mut o = 1.0;
    for i in 0..n_int {
        sum += o * noise_point(&(*p * lambda));
        lambda *= 1.99;
        o *= omega;
    }
    let n_partial = n - n_int as Float;
    sum += o * smooth_step(0.3, 0.7, n_partial) * noise_point(&(*p * lambda));
    sum
}

pub fn turbulence(
    p: &Point3f,
    dpdx: &Vector3f,
    dpdy: &Vector3f,
    omega: Float,
    max_octaves: Float,
) -> Float {
    let len2 = dpdx.length_squared().max(dpdy.length_squared());
    let n = clamp(-1.0 - 0.5 * len2.log2(), 0.0, max_octaves);
    let n_int = n.floor() as i32;

    let mut sum = 0.0;
    let mut lambda = 1.0;
    let mut o = 1.0;
    for i in 0..n_int {
        sum += o * noise_point(&(*p * lambda)).abs();
        lambda *= 1.99;
        o *= omega;
    }
    let n_partial = n - n_int as Float;
    sum += o * lerp(
        smooth_step(0.3, 0.7, n_partial),
        0.2,
        noise_point(&(*p * lambda).abs()),
    );
    for i in n_int..max_octaves as i32 {
        sum += o * 0.2;
        o *= omega;
    }
    sum
}

#[inline]
fn smooth_step(min: Float, max: Float, value: Float) -> Float {
    let v = clamp((value - min) / (max - min), 0.0, 1.0);
    v * v * (-2.0 * v + 3.0)
}

#[inline]
fn grad(x: i32, y: i32, z: i32, dx: Float, dy: Float, dz: Float) -> Float {
    let mut h = NOISE_PERM[NOISE_PERM[NOISE_PERM[x as usize] + y as usize] + z as usize];
    h &= 15;
    let u = if h < 8 || h == 12 || h == 13 { dx } else { dy };
    let v = if h < 4 || h == 12 || h == 13 { dy } else { dz };
    let u = if (h & 1) != 0 { -u } else { u };
    let v = if (h & 2) != 0 { -v } else { v };
    u + v
}

#[inline]
fn noise_weight(t: Float) -> Float {
    let t3 = t * t * t;
    let t4 = t3 * t;
    6.0 * t4 * t - 15.0 * t4 + 10.0 * t3
}

const NOISE_PERM_SIZE: usize = 256;
const NOISE_PERM: [usize; 2 * NOISE_PERM_SIZE] = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69,
    142, // Remainder of the noise permutation table
    8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203,
    117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74,
    165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220,
    105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132,
    187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3,
    64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59,
    227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70,
    221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232,
    178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162,
    241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204,
    176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141,
    128, 195, 78, 66, 215, 61, 156, 180, 151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194,
    233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234,
    75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174,
    20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83,
    111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25,
    63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188,
    159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147,
    118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170,
    213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253,
    19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193,
    238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31,
    181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
];
