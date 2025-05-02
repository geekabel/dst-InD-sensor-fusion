# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5] - 2025-04-29

### Added

- **Configuration File:** Introduced `config.ini` for easier configuration of the inD dataset path.
- **Path Configuration Logic:** Added `configparser` support in `main.py` to read `dataset_path` from `config.ini`. The path priority is now: Command-Line Argument > `config.ini` > Default Path.

### Changed

- Refactored `main.py` to use the new path configuration logic.

## [1.4] - 2025-04-29

### Fixed

- **`MapSensor` Attribute Check:** Fixed the `AttributeError: 'Lanelet' object has no attribute 'hasAttribute'` in `MapSensor._get_lanelet_subtype`. Replaced the incorrect `hasAttribute("subtype")` call with the correct Python dictionary-style check `"subtype" in nearest_lanelet.attributes`.

## [1.3] - 2025-04-29

### Fixed

- **Lanelet2 Map Integration:** Successfully implemented Lanelet2 map loading using `LocalCartesianProjector` with the correct latitude/longitude origin obtained from the recording metadata (`latLocation`, `lonLocation`). The `MapSensor` now uses actual map context instead of falling back to simplified logic.
- Fixed parameter name mismatches during sensor initialization in `main.py` (e.g., `position_noise_std_dev` for `CameraSensor`).
- Fixed various minor bugs and import errors encountered during testing (e.g., `tqdm` import, f-string syntax, helper function calls, metric calculation logic, visualization function names, syntax errors in `plotter.py`).

### Added

- Added `LocalCartesianProjector` logic to `main.py` for map loading.
- Added `plot_metrics_comparison` function to `visualization/plotter.py`.
- Added `tqdm` for progress bar during frame processing.
- Added more detailed verbose output and error handling during map loading.

## [1.2] - 2025-04-29

### Changed

- Updated documentation (`README.md`, `CHANGE.md`, main documentation) to reflect the status of `MapSensor` (falling back to simplified logic due to unresolved Lanelet2 projection issues).
- Cleaned up debug print statements and temporary files from v1.1 testing.

### Fixed

- Corrected map file path construction in `main.py` to point to `maps/lanelets/location_X/location<X>.osm`.
- Investigated Lanelet2 projection errors (`Latitude not in [-90d, 90d]`) related to `UtmProjector` and coordinate system mismatch.

## [1.1] - 2025-04-28

### Added

- Implemented `MapSensor` using `lanelet2` library to attempt loading actual map data.
- Added fallback mechanism in `MapSensor` to use simplified logic if map loading fails.
- Added realistic noise models to `CameraSensor`, `RadarSensor`, and `LidarSensor`:
  - Camera: Gaussian noise on position.
  - Radar: Differential noise (range/lateral) on position, noise on velocity.
  - Lidar: Gaussian noise on position.
- Added `README.md` and `CHANGE.md` files.
- Added command-line arguments to `main.py` for configuration.
- Added `tqdm` progress bar.

### Changed

- Refactored `main.py` to include map loading and pass map data/projector to `MapSensor`.
- Updated main documentation to reflect new sensor models and map integration attempt.

## [1.0] - 2025-04-28

### Added

- Initial implementation of the sensor fusion system.
- Core DST logic (`dst_core/basic.py`) with `MassFunction` and combination rules (Dempster, Yager, PCR5).
- Basic virtual sensors (`sensors/virtual_sensors.py`) with simplified logic and noise.
- Data loading modules (`data_loader/loader.py`) for inD dataset CSVs.
- Classification module (`classification.py`) with pignistic decision and metrics.
- Visualization tools (`visualization/plotter.py`) for BBAs and belief/plausibility.
- Main script (`main.py`) to run the experiment.

## [1.5] - 2025-04-29

### Added

- Added `config.ini` file for easier configuration of the inD dataset path.
- Conducted expanded testing on recordings 10, 20, and 30.
- Added analysis of expanded test results to `sensor_fusion_dst_documentation.md`.

### Changed

- Modified `main.py` to read dataset path from `config.ini` or command-line argument.
- Updated `README.md` and `sensor_fusion_dst_documentation.md` to explain the new configuration method and include expanded test analysis.
