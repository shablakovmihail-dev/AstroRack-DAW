# Changelog

All notable changes to AstroRack DAW are documented in this file.

## [v8.7.6] - 2026-11-11

### Added
- **Lunar Crater Scale Calibration**: Calibration tool using landmark lunar craters (Tycho, Copernicus, Plato, Clavius, etc.) with real physical diameters from `core/constants.py`.
- **Unified Metric Scale (`scale_km_per_px`)**: Unified coordinate and distance scaling across canvas measurement tools, crater ellipses, and lunar rulers.

### Fixed
- **Ellipse Render Crash**: Fixed crash during elliptical crater fit rendering on high-resolution/4K lunar surfaces.
- **Pipeline & Cache Stability**: Optimized 4K DSP pipeline, state LRU caching, and smooth wheel scroll UX.

## [v8.3.6] - 2026-10-30

### Added
- **Standalone Executable Packaging**: PyInstaller bundle configuration with pre-packaged astronomy sample stacks.
- **Runtime Asset Resolution**: Safe resource path resolution via `sys._MEIPASS` and `resource_path()`.
