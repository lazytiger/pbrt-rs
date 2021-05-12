use crate::core::{
    geometry::{Point2f, Vector2f, Vector3f},
    pbrt::{
        clamp, find_interval, Float, INV_2_PI, INV_4_PI, INV_PI, ONE_MINUS_EPSILON, PI, PI_OVER_2,
        PI_OVER_4,
    },
    rng::RNG,
};
use std::ops::IndexMut;

pub fn stratified_sample_1d(samples: &mut [Float], n_samples: usize, rng: &mut RNG, jitter: bool) {
    let inv_n_samples = 1.0 / n_samples as Float;
    for i in 0..n_samples {
        let delta = if jitter { rng.uniform_float() } else { 0.5 };
        samples[i] = ONE_MINUS_EPSILON.min((i as Float + delta) * inv_n_samples);
    }
}

pub fn stratified_sample_2d(
    samples: &mut [Point2f],
    nx: usize,
    ny: usize,
    rng: &mut RNG,
    jitter: bool,
) {
    let dx = 1.0 / nx as Float;
    let dy = 1.0 / ny as Float;
    let mut i = 0;
    for y in 0..ny {
        for x in 0..nx {
            let (jx, jy) = if jitter {
                (rng.uniform_float(), rng.uniform_float())
            } else {
                (0.5, 0.5)
            };
            samples[i].x = ONE_MINUS_EPSILON.min((x as Float + jx) * dx);
            samples[i].y = ONE_MINUS_EPSILON.min((y as Float + jy) * dy);
            i += 1;
        }
    }
}

// n_dim: represent dimensions of T
pub fn latin_hyper_cube<T: IndexMut<usize, Output = Float>>(
    samples: &mut [T],
    n_samples: usize,
    n_dim: usize,
    rng: &mut RNG,
) {
    let inv_n_samples = 1.0 / n_samples as Float;
    for i in 0..n_samples {
        for j in 0..n_dim {
            let sj = (i as Float + rng.uniform_float()) * inv_n_samples;
            samples[n_dim * i][j] = ONE_MINUS_EPSILON.min(sj);
        }
    }

    for i in 0..n_dim {
        for j in 0..n_samples {
            let other = j + rng.uniform_u32_u32((n_samples - j) as u32) as usize;
            let tmp = samples[n_dim * j][i];
            samples[n_dim * j][i] = samples[n_dim * other][i];
            samples[n_dim * other][i] = tmp;
        }
    }
}

#[derive(Debug)]
pub struct Distribution1D {
    pub(crate) func: Vec<Float>,
    cdf: Vec<Float>,
    pub(crate) func_int: Float,
}

impl Distribution1D {
    pub fn new(f: &[Float]) -> Distribution1D {
        let n = f.len();
        let mut d = Distribution1D {
            func: f.into(),
            cdf: Vec::with_capacity(n + 1),
            func_int: 0.0,
        };
        d.cdf[0] = 0.0;
        for i in 1..n + 1 {
            d.cdf[i] = d.cdf[i - 1] + d.func[i - 1] / n as Float;
        }
        d.func_int = d.cdf[n];
        if d.func_int == 0.0 {
            for i in 1..n + 1 {
                d.cdf[i] = i as Float / n as Float;
            }
        } else {
            for i in 1..n + 1 {
                d.cdf[i] /= d.func_int;
            }
        }
        d
    }

    pub fn count(&self) -> usize {
        self.func.len()
    }

    pub fn sample_continuous(
        &self,
        u: Float,
        pdf: Option<&mut Float>,
        off: Option<&mut usize>,
    ) -> Float {
        let offset = find_interval(self.cdf.len(), |index| self.cdf[index] < u);
        if let Some(off) = off {
            *off = offset;
        }
        let mut du = u - self.cdf[offset];
        if self.cdf[offset + 1] - self.cdf[offset] > 0.0 {
            du /= self.cdf[offset + 1] - self.cdf[offset];
        }

        if let Some(pdf) = pdf {
            *pdf = if self.func_int > 0.0 {
                self.func[offset] / self.func_int
            } else {
                0.0
            };
        }

        (offset as Float + du) / self.count() as Float
    }

    pub fn sample_discrete(
        &self,
        u: Float,
        pdf: Option<&mut Float>,
        u_remapped: Option<&mut Float>,
    ) -> usize {
        let offset = find_interval(self.cdf.len(), |index| self.cdf[index] < u);
        if let Some(pdf) = pdf {
            *pdf = if self.func_int > 0.0 {
                self.func[offset] / (self.func_int * self.count() as Float)
            } else {
                0.0
            }
        }

        if let Some(u_remapped) = u_remapped {
            *u_remapped = (u - self.cdf[offset]) / (self.cdf[offset + 1] - self.cdf[offset]);
        }
        offset
    }

    pub fn discrete_pdf(&self, index: usize) -> Float {
        self.func[index] / (self.func_int * self.count() as Float)
    }
}

pub fn rejection_sample_disk(rng: &mut RNG) -> Point2f {
    let mut p = Point2f::default();
    loop {
        p.x = 1.0 - 2.0 * rng.uniform_float();
        p.y = 1.0 - 2.0 * rng.uniform_float();
        if p.x * p.x + p.y * p.y <= 1.0 {
            break;
        }
    }
    p
}

#[derive(Debug)]
pub struct Distribution2D {
    conditional: Vec<Distribution1D>,
    marginal: Distribution1D,
}

impl Distribution2D {
    pub fn new(data: &[Float], nu: usize, nv: usize) -> Distribution2D {
        let mut conditional = Vec::with_capacity(nv);
        for v in 0..nv {
            conditional.push(Distribution1D::new(&data[v * nv..v * nv + nu]));
        }
        let mut marginal_func = Vec::with_capacity(nv);
        for v in 0..nv {
            marginal_func.push(conditional[v].func_int);
        }
        let marginal = Distribution1D::new(marginal_func.as_slice());
        Self {
            conditional,
            marginal,
        }
    }

    pub fn sample_continuous(&self, u: Point2f, pdf: Option<&mut Float>) -> Point2f {
        let mut pdfs = [0.0; 2];
        let mut v = 0;
        let d1 = self
            .marginal
            .sample_continuous(u[1], Some(&mut pdfs[1]), Some(&mut v));
        let d0 = self.conditional[v].sample_continuous(u[0], Some(&mut pdfs[0]), None);
        if let Some(pdf) = pdf {
            *pdf = pdfs[0] * pdfs[1];
        }
        Point2f::new(d0, d1)
    }

    pub fn pdf(&self, p: &Point2f) -> Float {
        let iu = clamp(
            p[0] as isize * self.conditional[0].count() as isize,
            0,
            self.conditional[0].count() as isize - 1,
        ) as usize;
        let iv = clamp(
            p[1] as isize * self.marginal.count() as isize,
            0,
            self.marginal.count() as isize - 1,
        ) as usize;
        self.conditional[iv].func[iu] / self.marginal.func_int
    }
}

pub fn uniform_sample_hemisphere(u: &Point2f) -> Vector3f {
    let z = u[0];
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = 2.0 * PI * u[1];
    Vector3f::new(r * phi.cos(), r * phi.sin(), z)
}

pub fn uniform_hemisphere_pdf() -> Float {
    INV_2_PI
}

pub fn uniform_sample_sphere(u: &Point2f) -> Vector3f {
    let z = 1.0 - 2.0 * u[0];
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = 2.0 * PI * u[1];
    Vector3f::new(r * phi.cos(), r * phi.sin(), z)
}

pub fn uniform_sphere_pdf() -> Float {
    INV_4_PI
}

pub fn uniform_sample_clone(u: &Point2f, theta_max: Float) -> Vector3f {
    let cos_theta = (1.0 - u[0]) + u[0] * theta_max;
    let sin_theta = (1.0 - theta_max * theta_max).sqrt();
    let phi = u[1] * 2.0 * PI;
    Vector3f::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta)
}

pub fn uniform_cone_pdf(cos_theta_max: Float) -> Float {
    1.0 / (2.0 * PI * (1.0 - cos_theta_max))
}

pub fn uniform_sample_disk(u: &Point2f) -> Point2f {
    let r = u[0].sqrt();
    let theta = 2.0 * PI * u[1];
    Point2f::new(r * theta.cos(), r * theta.sin())
}

pub fn concentric_sample_disk(u: &Point2f) -> Point2f {
    let u_offset = *u * 2.0 - Vector2f::new(1.0, 1.0);
    if u_offset.x == 0.0 && u_offset.y == 0.0 {
        return Point2f::default();
    }

    let (r, theta) = if u_offset.x.abs() > u_offset.y.abs() {
        (u_offset.x, PI_OVER_4 * (u_offset.y / u_offset.x))
    } else {
        (
            u_offset.y,
            PI_OVER_2 - PI_OVER_4 * (u_offset.x / u_offset.y),
        )
    };
    Point2f::new(theta.cos(), theta.sin()) * r
}

pub fn uniform_sample_triangle(u: &Point2f) -> Point2f {
    let su0 = u[0].sqrt();
    Point2f::new(1.0 - su0, u[1] * su0)
}

pub fn shuffle<T>(samp: &mut [T], count: usize, n_dimensions: usize, rng: &mut RNG) {
    for i in 0..count {
        let other = i + rng.uniform_u32_u32((count - i) as u32) as usize;
        for j in 0..n_dimensions {
            samp.swap(n_dimensions * i + j, n_dimensions * other + j);
        }
    }
}

#[inline]
pub fn cosine_sample_hemisphere(u: &Point2f) -> Vector3f {
    let d = concentric_sample_disk(u);
    let z = (1.0 - d.x * d.x - d.y * d.y).max(0.0);
    Vector3f::new(d.x, d.y, z)
}

#[inline]
pub fn cosine_hemisphere_pdf(cos_theta: Float) -> Float {
    cos_theta * INV_PI
}

#[inline]
pub fn balance_heuristic(nf: usize, f_pdf: Float, ng: usize, g_pdf: Float) -> Float {
    (nf as Float * f_pdf) / (nf as Float * f_pdf + ng as Float * g_pdf)
}

#[inline]
pub fn power_heuristic(nf: usize, f_pdf: Float, ng: usize, g_pdf: Float) -> Float {
    let nf = nf as Float;
    let ng = ng as Float;
    let f = nf * f_pdf;
    let g = ng * g_pdf;
    (f * f) / (f * f + g * g)
}
