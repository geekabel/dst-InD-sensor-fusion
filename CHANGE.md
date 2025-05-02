# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2025-04-28

### Added
- Realistic noise models to virtual sensors:
  - `CameraSensor`: Gaussian noise added to position.
  - `RadarSensor`: Differential noise (range/lateral) added to position, Gaussian noise to velocity.
  - `LidarSensor`: Gaussian noise added to position.
- `README.md` file with installation, usage, and package details.
- `CHANGE.md` file (this file).

### Changed
- `MapSensor` implementation:
  - Now uses the `lanelet2` library to query actual map data (.osm files) from the inD dataset.
  - Determines lanelet subtype based on object position and assigns belief accordingly.
  - Includes fallback to simplified logic if `lanelet2` is not installed.
- Updated `main.py` to load the Lanelet2 map and pass it to the `MapSensor`.
- Updated documentation (`sensor_fusion_dst_documentation.md`) to reflect sensor model changes and Lanelet2 integration.

## [1.0.0] - 2025-04-28

### Added
- Initial implementation of Dempster-Shafer sensor fusion for object classification.
- Core DST logic (`dst_core`) including `MassFunction`, combination rules (Dempster, Yager, PCR5), discounting, and decision support.
- Data loader (`data_loader`) for inD dataset CSV files.
- Virtual sensor simulations (`sensors`) for Camera, Radar, Lidar, and Map (simplified).
- Classification module (`classification`) with evaluation metrics.
- Visualization tools (`visualization`) using Matplotlib.
- Main experiment script (`main.py`) to run tests and compare combination rules.
- Initial documentation (`sensor_fusion_dst_documentation.md`).
- Test results and plots in `/results` directory.
