"""Астрономические константы и справочные данные для AstroRack DAW."""

from typing import Dict
from version import __version__, VERSION

# Справочник эталонных лунных кратеров с физическими диаметрами в километрах
LUNAR_CRATERS: Dict[str, float] = {
    "Tycho": 85.0,
    "Copernicus": 93.0,
    "Plato": 101.0,
    "Clavius": 225.0,
    "Theophilus": 100.0,
    "Aristarchus": 40.0,
    "Kepler": 31.0,
    "Langrenus": 132.0,
    "Gassendi": 110.0,
}

DEFAULT_LUNAR_DISTANCE_KM: float = 384400.0
