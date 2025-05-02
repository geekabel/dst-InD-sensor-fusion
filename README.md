# Sensor Fusion using Dempster-Shafer Theory for Object Classification

This package implements a sensor fusion system using Dempster-Shafer Theory (DST) for object classification based on the inD dataset. It simulates multiple virtual sensors (Camera, Radar, Lidar, Map) with realistic noise models, generates belief functions for object classification, and fuses them using various DST combination rules.

## Features

- **Core Dempster-Shafer Theory Implementation**:

  - Basic Belief Assignment (BBA) representation
  - Multiple combination rules: Dempster's rule, Yager's rule, PCR5
  - Belief, plausibility, and pignistic probability calculations
  - Classical and contextual discounting

- **Realistic Sensor Simulation**:

  - Camera sensor with Gaussian position noise
  - Radar sensor with differential noise (better range, worse lateral accuracy)
  - Lidar sensor with typical position noise
  - Map sensor using actual Lanelet2 map data

- **Classification and Evaluation**:

  - Decision-making based on maximum belief, plausibility, or pignistic probability
  - Performance metrics: accuracy, precision, recall, F1-score, confusion matrix

- **Visualization**:
  - Mass function visualization
  - Belief-plausibility interval plots
  - Performance comparison plots

## Installation

### Prerequisites

- Python 3.6+
- NumPy
- Pandas
- Matplotlib
- Lanelet2 (for map data processing)

### Setup

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/sensor_fusion_dst.git
   cd sensor_fusion_dst
   ```

2. Install dependencies:

   ```
   pip install numpy pandas matplotlib lanelet2
   ```

3. Prepare the inD dataset:
   - Download the inD dataset from [https://www.ind-dataset.com/](https://www.ind-dataset.com/)
   - Extract the dataset to a known location

## Usage

### Basic Usage

```python
from dst_core.basic import MassFunction
from sensors.virtual_sensors import CameraSensor, RadarSensor, LidarSensor, MapSensor
from classification import make_decision
import lanelet2

# Define frame of discernment
frame = {"car", "truck_bus", "bicycle", "pedestrian", "unknown"}

# Create sensors
camera = CameraSensor(frame, reliability=0.9, position=(0, 0), position_noise_std_dev=0.5)
radar = RadarSensor(frame, reliability=0.8, position=(0, 0),
                   range_noise_std_dev=0.3, lateral_noise_std_dev=1.0, velocity_noise_std_dev=0.2)
lidar = LidarSensor(frame, reliability=0.85, position=(0, 0), position_noise_std_dev=0.2)
map_sensor = MapSensor(frame, reliability=0.7)

# Load Lanelet2 map
projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(49.0, 8.0))
lanelet_map = lanelet2.io.load("path/to/map.osm", projector)

# Generate BBAs from sensors
camera_bba = camera.generate_classification_bba(track_data, track_meta)
radar_bba = radar.generate_classification_bba(track_data, track_meta)
lidar_bba = lidar.generate_classification_bba(track_data, track_meta)
map_bba = map_sensor.generate_classification_bba(track_data, track_meta,
                                               lanelet_map=lanelet_map, projector=projector)

# Combine BBAs using Dempster's rule
fused_bba = camera_bba.combine_dempster(radar_bba).combine_dempster(lidar_bba).combine_dempster(map_bba)

# Make decision based on maximum pignistic probability
decision = make_decision(fused_bba, method="max_pignistic")
```

### Running the Main Experiment

```bash
python main.py --dataset_path /path/to/inD-dataset --recording_id 1 --num_frames 50 --frame_step 10
```

## Package Structure

- `dst_core/`: Core Dempster-Shafer Theory implementation
  - `basic.py`: Basic belief functions, combination rules, and decision support
- `data_loader/`: Data loading utilities for the inD dataset
  - `loader.py`: Functions to load and process inD dataset files
- `sensors/`: Virtual sensor implementations
  - `virtual_sensors.py`: Camera, Radar, Lidar, and Map sensor simulations
- `visualization/`: Visualization tools
  - `plotter.py`: Functions for plotting mass functions and belief-plausibility intervals
- `classification.py`: Decision-making and evaluation metrics
- `main.py`: Main experiment script

## Sensor Models

### Camera Sensor

- Adds Gaussian noise to position measurements
- Classification confidence decreases with distance and for smaller objects
- Higher confidence for cars/trucks, potential confusion between bicycles/pedestrians

### Radar Sensor

- Adds differential noise to position (better range accuracy, poorer lateral accuracy)
- Adds noise to velocity measurements
- Limited classification ability, mainly distinguishes between vehicles and VRUs

### Lidar Sensor

- Adds typical lidar noise to position measurements
- Classification based on size and shape
- Moderate classification ability

### Map Sensor

- Uses Lanelet2 map data to determine the context (road, sidewalk, bicycle lane, etc.)
- Assigns belief based on typical road usage patterns for each context

## Results

The system was tested on the inD dataset with different combination rules (Dempster, Yager, PCR5). Results show high classification accuracy, with detailed performance metrics available in the `/results` directory after running the experiment.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The inD dataset: [https://levelxdata.com/ind-dataset/](https://levelxdata.com/ind-dataset/)
- Lanelet2: [https://github.com/fzi-forschungszentrum-informatik/Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2)
