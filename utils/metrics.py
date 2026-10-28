#"""Реэкспорт аналитических метрик из core.analysis для совместимости импортов."""
from core.analysis import (
    AnalysisResult,
    calculate_fwhm,
    calculate_angular_scale,
    FWHM_TO_SIGMA,
    SIGMA_TO_FWHM,
    ARCSEC_RAD_FACTOR,
)

__all__ = [
    "AnalysisResult",
    "calculate_fwhm",
    "calculate_angular_scale",
    "FWHM_TO_SIGMA",
    "SIGMA_TO_FWHM",
    "ARCSEC_RAD_FACTOR",
]
