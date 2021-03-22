struct CoefficientSpectrum {}

struct SampledSpectrum {}

struct RGBSpectrum {}

#[cfg(not(feature = "sampled_spectrum"))]
pub type Spectrum = RGBSpectrum;
#[cfg(feature = "sampled_spectrum")]
pub type Spectrum = SampledSpectrum;
