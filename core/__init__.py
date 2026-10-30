"""Ядро астрономической обработки изображений AstroRack."""
from version import __version__, VERSION
from .processing import AstroEngine, soft_clip_limiter
from .enhance import apply_local_contrast, apply_color_engine
from .pipeline import PipelineWorker
from .lunar_distance import calculate_lunar_distance, DEFAULT_LUNAR_DISTANCE_KM
from .metadata_reader import extract_capture_date

__all__ = [
    '__version__',
    'VERSION',
    'AstroEngine',
    'soft_clip_limiter',
    'PipelineWorker',
    'apply_local_contrast',
    'apply_color_engine',
    'calculate_lunar_distance',
    'DEFAULT_LUNAR_DISTANCE_KM',
    'extract_capture_date',
]
