The dataset provides processed information extracted from video recordings at specific urban/roundabout locations.
it consists of recording of recording from several German intersections capture via drone.

### Datasets components breakdown

## Recordings & Locations

Data is organized by `locationId` and split into continuous `recordingIds`.

## Core Data Files (per recording XX)

### XX_background.png

A static, georeferenced image of the scene. Useful for visualization and understanding the environment context.

### XX_recordingsMeta.csv

Metadata for the entire recording:

- `recordingId`, `locationId`: Identifiers.
- `frameRate`: Crucial for temporal calculations (e.g., time step = 1 / frameRate).
- `xUtmOrigin`, `yUtmOrigin`: Needed to convert local coordinates to global UTM coordinates.
- `orthoPxToMeter`: Scale factor for relating the background image pixels to meters (useful for visualization).
- Other fields provide context (time, duration, object counts, speed limit).

### XX_tracksMeta.csv

Metadata for each detected object track within the recording:

- `trackId`: Unique identifier for an object's trajectory within that recording.
- `initialFrame`, `finalFrame`, `numFrames`: Defines the temporal existence of the track.
- `class`: The ground truth classification (e.g., 'Car', 'Pedestrian', 'Bicyclist'). This is vital for evaluating our classification fusion. Note the grouping into VRUs (Vulnerable Road Users).
- `width`, `length`: Ground truth dimensions (often 0 for VRUs, which is a detail to note).

### XX_tracks.csv

The most important file containing the time-series data for each track. Each row represents an object's state at a specific frame:

- `trackId`, `frame`: Links the state to a specific object and time.
- `xCenter`, `yCenter`: Ground truth position in the local coordinate system [m].
- `heading`: Ground truth orientation in the local coordinate system [deg].
- `xVelocity`, `yVelocity`: Ground truth velocity components in the local coordinate system [m/s].
- `xAcceleration`, `yAcceleration`: Ground truth acceleration components in the local coordinate system [m/s].
- `lonVelocity`, `latVelocity`: Longitudinal/Lateral velocity relative to the object's heading [m/s].
- `lonAcceleration`, `latAcceleration`: Longitudinal/Lateral acceleration [m/s²].
- `width`, `length`: Dimensions at that frame (often 0 for VRUs).

## Map Data (per location)- german intersection

- **Lanelet2 (.osm) and OpenDRIVE (.xodr)**: Rich, structured map information including lane boundaries, types (road, sidewalk, bike lane), connections, traffic rules (speed limits via speed_limit tag, potential stop lines, etc.). Crucial for contextual reasoning.
- **3D Scene (.osgb & .fbx)**: For detailed 3D visualization if needed.

## Coordinate System

### Local System

The primary system for XX_tracks.csv data (xCenter, yCenter, heading, velocities, accelerations):

- Origin (0,0) is fixed per location, close to the intersection.
- X-axis grows right, Y-axis grows upwards.
- Heading is likely measured counter-clockwise from the positive X-axis (standard mathematical convention, but verify if possible, e.g., by visualizing a car moving along the X-axis).

### Global UTM

Convert local positions using:

- xUtm = xCenter + xUtmOrigin
- yUtm = yCenter + yUtmOrigin

## Handling the Data for Dempster-Shafer Experiments

The idea is to use this dataset as ground truth to simulate multiple, potentially uncertain or conflicting, "virtual sensors." we will then apply DST to fuse the information from these simulated sensors and compare the result against the known ground truth.

### Workflow

_Adjust it when I'm coding bro_

#### 1. Load Data

- Read the XX_recordingsMeta.csv to get frameRate, xUtmOrigin, yUtmOrigin, etc.
- Read the XX_tracksMeta.csv to get the list of tracks, their classes, and lifetimes.
- Read the XX_tracks.csv into a suitable structure (e.g., a dictionary mapping trackId to a list/dataframe of states over time, or a large dataframe indexed by trackId and frame).
- Load the corresponding Lanelet2 map (.osm) for the locationId. Libraries like lanelet2 (Python) can parse these.

#### 2. Define the Frame of Discernment (Ω)

The task, we are going to index here -> _classification_
For example in this workflow, we will define this

- **Object Classification**: Ω = {Car, Truck, Bus, Pedestrian, Bicyclist, Motorcycle, Unknown} (align with classes present in XX_tracksMeta.csv).

#### 3. Simulate Virtual Sensors

For each trackId at each frame:

1. Access the ground truth state (position, velocity, class, etc.) from XX_tracks.csv and XX_tracksMeta.csv.
2. Access map context using the object's position (xCenter, yCenter) and the loaded Lanelet2 map.
3. Generate Belief/Mass Functions (BBAs) for each simulated sensor based on the ground truth, adding realistic noise and uncertainty:

##### Simulated Camera

- **Classification**: Generate a BBA for classification. High confidence if the object class is 'Car' or 'Truck'. Lower confidence, perhaps confusion between 'Bicyclist' and 'Motorcycle', or 'Pedestrian' and 'Bicyclist' if they are close/small. Add noise (e.g., small chance of misclassification). m({CorrectClass}) = 0.7, m({ConfusedClass}) = 0.1, m({CorrectClass, ConfusedClass}) = 0.1, m(Ω) = 0.1.
- **Position**: Use xCenter, yCenter but add Gaussian noise. The uncertainty (m(Ω)) could increase with distance from a hypothetical camera origin or for smaller objects.

##### Simulated Radar

- **Position/Velocity**: Good at range and radial velocity. Simulate based on xCenter, yCenter, lonVelocity. Add higher noise to lateral position/velocity (yCenter relative to radar, latVelocity).
- **Classification**: Poor classification ability. Generate a BBA with high mass on larger sets or 'Unknown'. m({Vehicle}) = 0.5, m({VRU}) = 0.2, m(Ω) = 0.3. (Where Vehicle = {Car, Truck, Bus}, VRU = {Pedestrian, Bicyclist, Motorcycle}).

##### Simulated Lidar

- **Position/Shape**: Good at position and detecting presence/shape. Use xCenter, yCenter, potentially width/length (handle the zero values for VRUs - perhaps assign a default small dimension). Add Gaussian noise to position.
- **Classification**: Limited. Maybe differentiate between large shapes (Vehicles) and small shapes (VRUs). Generate BBAs reflecting this.

##### Map-Based "Sensor"

- **Classification/Location Context**: Use the object's xCenter, yCenter and the Lanelet2 map. If the object is within a lanelet tagged as 'road', increase belief for {Car, Truck, Bus, Motorcycle}. If on a 'sidewalk', increase belief for {Pedestrian}. If on a 'bicycle_lane', increase belief for {Bicyclist}. Generate BBAs reflecting this prior knowledge. m({Pedestrian}) = 0.6 if on sidewalk, m(Ω) = 0.4.

#### 4. Apply Dempster's Rule of Combination

- For each object at each frame, take the BBAs generated by your simulated sensors (e.g., Camera BBA, Radar BBA, Map BBA).
- Combine them using Dempster's rule to get a fused BBA. Handle potential high conflict between sources.

#### 5. Decision Making

- Based on the fused BBA, make a decision. For classification, you might choose the hypothesis with the highest Belief (Bel) or Plausibility (Pl).

#### 6. Evaluation

- Use and Compare some decision against the ground truth from the dataset.
- For classification: Compare the chosen class against the class in XX_tracksMeta.csv.
- Calculate performance metrics (accuracy, confusion matrix, precision, recall, F1-score) over all objects and frames.

### Example Scenario (Classification)

- **Frame t, Track k**: Ground truth class is 'Bicyclist'.
- **Simulated Camera**: Sees a small, moving object. BBA: m({Bicyclist})=0.6, m({Pedestrian})=0.2, m({Bic, Ped})=0.1, m(Ω)=0.1.
- **Simulated Lidar**: Sees a small object shape. BBA: m({Bicyclist, Pedestrian, Motorcycle})=0.7, m(Ω)=0.3. (Groups VRUs).
- **Map Sensor**: Object position is in a 'bicycle_lane'. BBA: m({Bicyclist})=0.8, m(Ω)=0.2.
- **Fusion**: Combine these three BBAs using Dempster's rule.
- **Decision**: The fused BBA will likely have the highest belief/plausibility for {Bicyclist}.
- **Evaluation**: Decision ('Bicyclist') matches ground truth ('Bicyclist'). Correct.
