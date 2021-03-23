use crate::Float;

struct CoefficientSpectrum {}

pub struct SampledSpectrum {}

pub struct RGBSpectrum {}

impl RGBSpectrum {
    pub fn new(f: Float) -> RGBSpectrum {
        RGBSpectrum {}
    }
}

#[cfg(not(feature = "sampled_spectrum"))]
pub type Spectrum = RGBSpectrum;
#[cfg(feature = "sampled_spectrum")]
pub type Spectrum = SampledSpectrum;
