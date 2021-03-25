use crate::core::geometry::{Point2f, Vector3f};
use crate::core::{clamp, find_interval};
use crate::{Float, PI};
use std::panic::PanicInfo;
use std::sync::Arc;

pub struct Distribution1D {
    func: Vec<Float>,
    cdf: Vec<Float>,
    func_int: Float,
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

pub fn uniform_sample_sphere(u: &Point2f) -> Vector3f {
    let z = 1.0 - 2.0 * u[0];
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = 2.0 * PI * u[1];
    Vector3f::new(r * phi.cos(), r * phi.sin(), z)
}

pub fn uniform_cone_pdf(cos_theta_max: Float) -> Float {
    1.0 / (2.0 * PI * (1.0 - cos_theta_max))
}