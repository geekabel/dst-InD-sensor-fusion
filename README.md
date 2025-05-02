# Sensor Fusion using Dempster-Shafer Theory with inD Dataset

This project implements a sensor fusion system using Dempster-Shafer Theory (DST) for object classification based on the [inD dataset](https://levelxdata.com/ind-dataset/). It simulates various sensors (Camera, Radar, Lidar, Map) with realistic noise models, generates Basic Belief Assignments (BBAs) from each sensor, fuses them using different combination rules (Dempster, Yager, PCR5), and performs classification based on the fused belief.

**Version:** 1.5

## Features

- **Dempster-Shafer Core:** Implements the `MassFunction` class with methods for combination (Dempster, Yager, PCR5), discounting, pignistic transformation, belief, and plausibility calculations.
- **Virtual Sensors:**
  - `CameraSensor`: Simulates classification based on distance and size, with Gaussian noise on position.
  - `RadarSensor`: Simulates classification based on size and speed, distinguishing between vehicles and VRUs, with differential noise (range/lateral) on position and noise on velocity.
  - `LidarSensor`: Simulates classification based on object dimensions (size, aspect ratio), with Gaussian noise on position.
  - `MapSensor`: **Successfully integrates Lanelet2 map data!** Uses `LocalCartesianProjector` with the correct origin from recording metadata to determine context (e.g., driving lane, intersection) based on the object's position within the loaded map.
- **Data Loading:** Modules to load recording metadata, track metadata, and track data from the inD dataset CSV files.
- **Classification:** Implements decision-making based on pignistic transformation and calculates standard classification metrics (accuracy, precision, recall, F1-score, confusion matrix).
- **Visualization:** Tools to plot mass functions, belief/plausibility intervals, and comparison of metrics across different combination rules.
- **Command-Line Interface:** A `main.py` script to run experiments with configurable parameters (dataset path, recording ID, frames, results directory, plotting, verbosity).
- **Configuration File:** A `config.ini` file for easier configuration of the inD dataset path.

## Installation

1. **Clone or download the package.**
2. **Install required Python libraries:**

   ```bash
   pip install pandas numpy matplotlib tqdm scikit-learn
   ```

3. **Install Lanelet2:** Follow the official installation instructions for your system: [https://github.com/fzi-forschungszentrum-informatik/Lanelet2#installation](https://github.com/fzi-forschungszentrum-informatik/Lanelet2#installation)
   - _Note:_ Ensure the Python bindings for Lanelet2 are correctly installed and accessible in your Python environment.
4. **Download the inD dataset:** Obtain the dataset from [https://www.ind-dataset.com/](https://www.ind-dataset.com/) and place it in a known location (e.g., `/path/to/inD_dataset`).
5. **Configure the dataset path:** Edit the `config.ini` file to specify the path to your inD dataset:

   ```ini
   [Paths]
   # Specify the root path to your downloaded inD dataset
   dataset_path = /path/to/inD_dataset
   ```

## Usage

Run the main script `main.py` from the `sensor_fusion_dst` directory:

```bash
python main.py --recording_id <id> [options]
```

**Arguments:**

- `--dataset_path`: Path to the root directory of the inD dataset (overrides the path in `config.ini`).
- `--recording_id`: The ID of the recording to process (e.g., 1, 2, ... default: 1).
- `--num_frames`: Number of frames to process from the start of the recording (default: 50).
- `--frame_step`: Step size between processed frames (default: 10).
- `--results_dir`: Directory to save output plots (default: `/home/ubuntu/results_v1.5`).
- `--plot`: Generate and save plots (mass functions, belief/plausibility, metrics comparison).
- `-v`, `--verbose`: Increase output verbosity (use `-vv` for more detail).
- `-q`, `--quiet`: Suppress progress bar and verbose output.

**Example:**

```bash
python main.py --recording_id 10 --num_frames 100 --frame_step 5 --results_dir ./results_recording10 --plot -v
```

This command will process recording 10 using the dataset path from `config.ini`, analyzing 100 frames with a step of 5, saving plots to `./results_recording10`, and providing verbose output.

## Project Structure

```
sensor_fusion_dst/
├── data_loader/          # Modules for loading inD data
│   ├── __init__.py
│   └── loader.py
├── dst_core/             # Core Dempster-Shafer implementation
│   ├── __init__.py
│   └── basic.py
├── sensors/              # Virtual sensor implementations
│   ├── __init__.py
│   └── virtual_sensors.py
├── visualization/        # Plotting utilities
│   ├── __init__.py
│   └── plotter.py
├── __init__.py
├── classification.py     # Classification logic and metrics
├── main.py               # Main executable script
├── config.ini            # Configuration file for dataset path
├── README.md             # This file
└── CHANGE.md             # Changelog
```

## Expanded Testing Results

The system has been tested on multiple recordings (10, 20, 30) from the inD dataset to evaluate performance across different scenarios:

- **Recording 10:** Achieved 0.864 accuracy with all combination rules. Only 'road' lanelet subtypes were encountered.
- **Recording 20:** Achieved perfect 1.0 accuracy despite encountering 'walkway' lanelet subtypes (flagged as unknown).
- **Recording 30:** Also achieved perfect 1.0 accuracy with 'walkway' lanelet subtypes present.

The variation in accuracy and the system's resilience to unknown lanelet subtypes demonstrate both the importance of diverse scenario testing and the robustness of the sensor fusion approach. See the full documentation for detailed analysis.

## Notes

- The `MapSensor` successfully uses Lanelet2 data. It determines the map origin (latitude, longitude) from the recording metadata and uses `LocalCartesianProjector`.
- The frame of discernment is currently fixed to `{"car", "van", "truck_bus", "pedestrian", "bicycle"}`. Other classes from the dataset are ignored.
- Sensor parameters (reliability, noise levels) are currently hardcoded in `main.py` but could be made configurable.
- The Lanelet2 map loading uses `loadRobust` to handle potential errors in the `.osm` files.
- The system currently flags 'walkway' lanelet subtypes as unknown. Future versions could extend the `MapSensor` to handle more lanelet subtypes explicitly.
