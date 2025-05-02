"""
Virtual sensor simulation for the inD dataset.

This module provides classes to simulate different types of sensors (camera, radar, lidar, map)
by adding realistic noise and uncertainty to the ground truth data from the inD dataset.
Each sensor generates Basic Belief Assignments (BBAs) for the classification task.

Refinement:
- MapSensor now uses actual Lanelet2 map data.
- Added realistic noise models to Camera, Radar, and Lidar sensors.
"""

import numpy as np
import pandas as pd
from typing import Dict, Set, FrozenSet, List, Tuple, Optional, Any
import sys
import os

# Add parent directory to path to import dst_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dst_core.basic import MassFunction, create_simple_mass_function

# Import Lanelet2
try:
    import lanelet2
    from lanelet2.core import BasicPoint2d
    from lanelet2.projection import UtmProjector
    from lanelet2.geometry import findNearest
    LANELET2_AVAILABLE = True
except ImportError:
    print("Warning: Lanelet2 library not found. MapSensor will use simplified logic.")
    LANELET2_AVAILABLE = False


class VirtualSensor:
    """Base class for all virtual sensors."""
    def __init__(self, frame_of_discernment: Set[str], reliability: float = 1.0):
        """
        Initialize a virtual sensor.
        Args:
            frame_of_discernment: Set of all possible classes
            reliability: Reliability factor for this sensor (0 to 1)
        """
        self.frame = frozenset(frame_of_discernment)
        if not 0 <= reliability <= 1:
            raise ValueError("Reliability must be between 0 and 1")
        self.reliability = reliability

    def generate_classification_bba(self, track_data: pd.Series,
                                   track_meta: pd.Series,
                                   **kwargs) -> MassFunction:
        """
        Generate a Basic Belief Assignment for classification.

        Args:
            track_data: Series containing track data for a specific frame
            track_meta: Series containing metadata for the track
            **kwargs: Additional arguments specific to the sensor (e.g., map data)

        Returns:
            A mass function representing the sensor's belief about the object class
        """
        raise NotImplementedError("Subclasses must implement this method")


class CameraSensor(VirtualSensor):
    """
    Simulates a camera sensor for object classification.

    Cameras are generally good at classification but performance decreases
    with distance and for smaller objects. Position estimation has noise.
    """

    def __init__(self, frame_of_discernment: Set[str],
                reliability: float = 0.9,
                position: Tuple[float, float] = (0, 0),
                position_noise_std_dev: float = 0.5): # meters
        """
        Initialize a camera sensor.

        Args:
            frame_of_discernment: Set of all possible classes
            reliability: Overall reliability factor for this sensor (0 to 1)
            position: (x, y) coordinates of the camera in the local coordinate system
            position_noise_std_dev: Standard deviation of Gaussian noise for position
        """
        super().__init__(frame_of_discernment, reliability)
        self.position = position
        self.position_noise_std_dev = position_noise_std_dev

    def generate_classification_bba(self, track_data: pd.Series,
                                   track_meta: pd.Series,
                                   **kwargs) -> MassFunction:
        """
        Generate a Basic Belief Assignment for classification based on camera simulation.

        Args:
            track_data: Series containing track data for a specific frame
            track_meta: Series containing metadata for the track

        Returns:
            A mass function representing the camera's belief about the object class
        """
        # --- Add Noise ---
        # Add Gaussian noise to position
        noise_x = np.random.normal(0, self.position_noise_std_dev)
        noise_y = np.random.normal(0, self.position_noise_std_dev)
        x_center_noisy = track_data["xCenter"] + noise_x
        y_center_noisy = track_data["yCenter"] + noise_y
        # -----------------

        # Use noisy position for calculations
        width = track_data["width"] # Assume size estimation is less affected by position noise
        length = track_data["length"]

        # Calculate distance from camera to noisy object position
        distance = np.sqrt((x_center_noisy - self.position[0])**2 +
                          (y_center_noisy - self.position[1])**2)

        # Calculate object size (area)
        size = width * length

        # Determine distance factor (decreases with distance)
        distance_factor = max(0.1, min(1.0, 1.0 - (distance / 100.0)))

        # Determine size factor (increases with size)
        size_factor = min(1.0, max(0.1, size / 0.5)) if size > 0 else 0.1

        # Combined factor affects the certainty of classification
        combined_factor = 0.7 * distance_factor + 0.3 * size_factor

        # Ground truth class (for simulation purposes)
        true_class = track_meta["class"]

        # Initialize mass dictionary
        mass_dict = {}

        # Simulate camera classification with realistic uncertainty
        if true_class == "car":
            correct_mass = 0.8 * combined_factor
            mass_dict[frozenset({"car"})] = correct_mass
            confusion_mass = 0.1 * combined_factor
            mass_dict[frozenset({"truck_bus"})] = confusion_mass
            uncertainty_mass = 0.05 * combined_factor
            mass_dict[frozenset({"car", "truck_bus"})] = uncertainty_mass

        elif true_class == "truck_bus":
            correct_mass = 0.8 * combined_factor
            mass_dict[frozenset({"truck_bus"})] = correct_mass
            confusion_mass = 0.1 * combined_factor
            mass_dict[frozenset({"car"})] = confusion_mass
            uncertainty_mass = 0.05 * combined_factor
            mass_dict[frozenset({"car", "truck_bus"})] = uncertainty_mass

        elif true_class == "bicycle":
            correct_mass = 0.7 * combined_factor
            mass_dict[frozenset({"bicycle"})] = correct_mass
            confusion_mass = 0.15 * combined_factor
            mass_dict[frozenset({"pedestrian"})] = confusion_mass
            uncertainty_mass = 0.1 * combined_factor
            mass_dict[frozenset({"bicycle", "pedestrian"})] = uncertainty_mass

        elif true_class == "pedestrian":
            correct_mass = 0.7 * combined_factor
            mass_dict[frozenset({"pedestrian"})] = correct_mass
            confusion_mass = 0.15 * combined_factor
            mass_dict[frozenset({"bicycle"})] = confusion_mass
            uncertainty_mass = 0.1 * combined_factor
            mass_dict[frozenset({"bicycle", "pedestrian"})] = uncertainty_mass

        # Remaining mass goes to complete uncertainty (the universal set)
        total_assigned_mass = sum(mass_dict.values())
        if total_assigned_mass < 1.0:
            mass_dict[self.frame] = 1.0 - total_assigned_mass

        # Create the mass function
        mass_function = MassFunction(self.frame, mass_dict)

        # Apply reliability discounting
        if self.reliability < 1.0:
            mass_function = mass_function.discount(self.reliability)

        return mass_function


class RadarSensor(VirtualSensor):
    """
    Simulates a radar sensor for object classification.

    Radars are generally poor at detailed classification but can distinguish
    between larger vehicles and smaller objects based on radar cross-section.
    They are good at detecting velocity, but position accuracy varies (better range, worse lateral).
    """

    def __init__(self, frame_of_discernment: Set[str],
                reliability: float = 0.8,
                position: Tuple[float, float] = (0, 0),
                range_noise_std_dev: float = 0.3, # meters
                lateral_noise_std_dev: float = 1.0, # meters
                velocity_noise_std_dev: float = 0.2): # m/s
        """
        Initialize a radar sensor.

        Args:
            frame_of_discernment: Set of all possible classes
            reliability: Overall reliability factor for this sensor (0 to 1)
            position: (x, y) coordinates of the radar in the local coordinate system
            range_noise_std_dev: Std dev of noise in the direction of the object
            lateral_noise_std_dev: Std dev of noise perpendicular to the object direction
            velocity_noise_std_dev: Std dev of noise for velocity components
        """
        super().__init__(frame_of_discernment, reliability)
        self.position = position
        self.range_noise_std_dev = range_noise_std_dev
        self.lateral_noise_std_dev = lateral_noise_std_dev
        self.velocity_noise_std_dev = velocity_noise_std_dev

    def generate_classification_bba(self, track_data: pd.Series,
                                   track_meta: pd.Series,
                                   **kwargs) -> MassFunction:
        """
        Generate a Basic Belief Assignment for classification based on radar simulation.

        Args:
            track_data: Series containing track data for a specific frame
            track_meta: Series containing metadata for the track

        Returns:
            A mass function representing the radar's belief about the object class
        """
        # --- Add Noise ---
        x_center_true = track_data["xCenter"]
        y_center_true = track_data["yCenter"]
        x_vel_true = track_data["xVelocity"]
        y_vel_true = track_data["yVelocity"]

        # Calculate relative position and angle
        rel_x = x_center_true - self.position[0]
        rel_y = y_center_true - self.position[1]
        angle = np.arctan2(rel_y, rel_x)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Generate noise in range and lateral directions
        range_noise = np.random.normal(0, self.range_noise_std_dev)
        lateral_noise = np.random.normal(0, self.lateral_noise_std_dev)

        # Convert noise to x, y coordinates
        noise_x = range_noise * cos_a - lateral_noise * sin_a
        noise_y = range_noise * sin_a + lateral_noise * cos_a

        x_center_noisy = x_center_true + noise_x
        y_center_noisy = y_center_true + noise_y

        # Add noise to velocity components (simplified isotropic noise)
        vel_noise_x = np.random.normal(0, self.velocity_noise_std_dev)
        vel_noise_y = np.random.normal(0, self.velocity_noise_std_dev)
        x_velocity_noisy = x_vel_true + vel_noise_x
        y_velocity_noisy = y_vel_true + vel_noise_y
        # -----------------

        # Use noisy data for calculations
        width = track_data["width"]
        length = track_data["length"]

        # Calculate distance from radar to noisy object position
        distance = np.sqrt((x_center_noisy - self.position[0])**2 +
                          (y_center_noisy - self.position[1])**2)

        # Calculate object size (area)
        size = width * length

        # Calculate noisy speed
        speed = np.sqrt(x_velocity_noisy**2 + y_velocity_noisy**2)

        # Determine distance factor (decreases with distance)
        distance_factor = max(0.2, min(1.0, 1.0 - (distance / 150.0)))

        # Determine size factor (increases with size)
        size_factor = min(1.0, max(0.1, size / 0.5)) if size > 0 else 0.1

        # Combined factor affects the certainty of classification
        combined_factor = 0.6 * distance_factor + 0.4 * size_factor

        # Ground truth class (for simulation purposes)
        true_class = track_meta["class"]

        # Initialize mass dictionary
        mass_dict = {}

        # Define vehicle and VRU sets
        vehicles = frozenset({"car", "truck_bus"})
        vrus = frozenset({"bicycle", "pedestrian"})

        # Radar can mainly distinguish between vehicles and VRUs
        if true_class in ["car", "truck_bus"]:
            vehicle_mass = 0.7 * combined_factor
            mass_dict[vehicles] = vehicle_mass
            vru_mass = 0.1 * combined_factor * (1.0 - size_factor)
            if vru_mass > 0.01:
                mass_dict[vrus] = vru_mass

        elif true_class in ["bicycle", "pedestrian"]:
            vru_mass = 0.6 * combined_factor
            mass_dict[vrus] = vru_mass
            vehicle_mass = 0.1 * combined_factor * (size_factor + min(1.0, speed / 10.0)) / 2.0
            if vehicle_mass > 0.01:
                mass_dict[vehicles] = vehicle_mass

        # Remaining mass goes to complete uncertainty (the universal set)
        total_assigned_mass = sum(mass_dict.values())
        if total_assigned_mass < 1.0:
            mass_dict[self.frame] = 1.0 - total_assigned_mass

        # Create the mass function
        mass_function = MassFunction(self.frame, mass_dict)

        # Apply reliability discounting
        if self.reliability < 1.0:
            mass_function = mass_function.discount(self.reliability)

        return mass_function


class LidarSensor(VirtualSensor):
    """
    Simulates a lidar sensor for object classification.

    Lidars are good at detecting object shapes and dimensions but
    have limited ability for detailed classification. Position estimation has noise.
    """

    def __init__(self, frame_of_discernment: Set[str],
                reliability: float = 0.85,
                position: Tuple[float, float] = (0, 0),
                position_noise_std_dev: float = 0.2): # meters (Lidar is generally more accurate)
        """
        Initialize a lidar sensor.

        Args:
            frame_of_discernment: Set of all possible classes
            reliability: Overall reliability factor for this sensor (0 to 1)
            position: (x, y) coordinates of the lidar in the local coordinate system
            position_noise_std_dev: Standard deviation of Gaussian noise for position
        """
        super().__init__(frame_of_discernment, reliability)
        self.position = position
        self.position_noise_std_dev = position_noise_std_dev

    def generate_classification_bba(self, track_data: pd.Series,
                                   track_meta: pd.Series,
                                   **kwargs) -> MassFunction:
        """
        Generate a Basic Belief Assignment for classification based on lidar simulation.

        Args:
            track_data: Series containing track data for a specific frame
            track_meta: Series containing metadata for the track

        Returns:
            A mass function representing the lidar's belief about the object class
        """
        # --- Add Noise ---
        # Add Gaussian noise to position
        noise_x = np.random.normal(0, self.position_noise_std_dev)
        noise_y = np.random.normal(0, self.position_noise_std_dev)
        x_center_noisy = track_data["xCenter"] + noise_x
        y_center_noisy = track_data["yCenter"] + noise_y
        # -----------------

        # Use noisy position for calculations
        width = track_data["width"]
        length = track_data["length"]

        # Calculate distance from lidar to noisy object position
        distance = np.sqrt((x_center_noisy - self.position[0])**2 +
                          (y_center_noisy - self.position[1])**2)

        # Calculate object size (area) and aspect ratio
        size = width * length
        aspect_ratio = length / width if width > 0 else 0

        # Determine distance factor (decreases with distance)
        distance_factor = max(0.2, min(1.0, 1.0 - (distance / 120.0)))

        # Determine size factor (increases with size)
        size_factor = min(1.0, max(0.1, size / 0.5)) if size > 0 else 0.1

        # Combined factor affects the certainty of classification
        combined_factor = 0.5 * distance_factor + 0.5 * size_factor

        # Ground truth class (for simulation purposes)
        true_class = track_meta["class"]

        # Initialize mass dictionary
        mass_dict = {}

        # Lidar classification based on size and shape
        if true_class == "car":
            car_mass = 0.6 * combined_factor
            mass_dict[frozenset({"car"})] = car_mass
            vehicle_mass = 0.2 * combined_factor
            mass_dict[frozenset({"car", "truck_bus"})] = vehicle_mass

        elif true_class == "truck_bus":
            truck_mass = 0.6 * combined_factor
            mass_dict[frozenset({"truck_bus"})] = truck_mass
            vehicle_mass = 0.2 * combined_factor
            mass_dict[frozenset({"car", "truck_bus"})] = vehicle_mass

        elif true_class == "bicycle":
            bicycle_mass = 0.5 * combined_factor
            mass_dict[frozenset({"bicycle"})] = bicycle_mass
            vru_mass = 0.3 * combined_factor
            mass_dict[frozenset({"bicycle", "pedestrian"})] = vru_mass

        elif true_class == "pedestrian":
            pedestrian_mass = 0.5 * combined_factor
            mass_dict[frozenset({"pedestrian"})] = pedestrian_mass
            vru_mass = 0.3 * combined_factor
            mass_dict[frozenset({"bicycle", "pedestrian"})] = vru_mass

        # Remaining mass goes to complete uncertainty (the universal set)
        total_assigned_mass = sum(mass_dict.values())
        if total_assigned_mass < 1.0:
            mass_dict[self.frame] = 1.0 - total_assigned_mass

        # Create the mass function
        mass_function = MassFunction(self.frame, mass_dict)

        # Apply reliability discounting
        if self.reliability < 1.0:
            mass_function = mass_function.discount(self.reliability)

        return mass_function


class MapSensor(VirtualSensor):
    """
    Simulates a map-based "sensor" for object classification using Lanelet2 map data.

    This represents prior knowledge from map data about where different types of
    road users are likely to be found.
    """

    def __init__(self, frame_of_discernment: Set[str], reliability: float = 0.7):
        """
        Initialize a map-based sensor.

        Args:
            frame_of_discernment: Set of all possible classes
            reliability: Overall reliability factor for this sensor (0 to 1)
        """
        super().__init__(frame_of_discernment, reliability)

        # Define probabilities based on lanelet subtypes
        self.subtype_probs = {
            "road": {"car": 0.7, "truck_bus": 0.2, "bicycle": 0.05, "pedestrian": 0.05},
            "highway": {"car": 0.8, "truck_bus": 0.2, "bicycle": 0.0, "pedestrian": 0.0},
            "bicycle_lane": {"car": 0.05, "truck_bus": 0.0, "bicycle": 0.85, "pedestrian": 0.1},
            "sidewalk": {"car": 0.01, "truck_bus": 0.0, "bicycle": 0.1, "pedestrian": 0.89},
            "crosswalk": {"car": 0.05, "truck_bus": 0.0, "bicycle": 0.2, "pedestrian": 0.75},
            "parking": {"car": 0.8, "truck_bus": 0.1, "bicycle": 0.05, "pedestrian": 0.05},
            "unknown": {"car": 0.25, "truck_bus": 0.15, "bicycle": 0.3, "pedestrian": 0.3}
        }

    def _get_lanelet_subtype(self, x: float, y: float,
                             lanelet_map: Optional[Any], # lanelet2.core.LaneletMap
                             projector: Optional[Any] # lanelet2.projection.Projector
                             ) -> str:
        """
        Determine the subtype of the lanelet at the given local coordinates.

        Args:
            x: x-coordinate in the local coordinate system
            y: y-coordinate in the local coordinate system
            lanelet_map: Loaded Lanelet2 map object.
            projector: Lanelet2 projector object.

        Returns:
            Lanelet subtype as string (e.g., "road", "sidewalk") or "unknown".
        """
        # # --- DEBUG ---
        # print(f"DEBUG (MapSensor): _get_lanelet_subtype called for ({x}, {y})")
        # print(f"DEBUG (MapSensor): LANELET2_AVAILABLE = {LANELET2_AVAILABLE}")
        # print(f"DEBUG (MapSensor): lanelet_map is None = {lanelet_map is None}")
        # print(f"DEBUG (MapSensor): projector is None = {projector is None}")
        # # --- END DEBUG ---

        if not LANELET2_AVAILABLE or lanelet_map is None or projector is None:
            print("Warning (MapSensor): Using simplified map logic.") # Explicitly state source
            # Simplified logic based on y-coordinate (as before)
            if y < -5 or y >= 5:
                return "sidewalk"
            elif (-5 <= y < -2) or (2 <= y < 5):
                return "bicycle_lane"
            elif -2 <= y < 2:
                return "road"
            else:
                return "unknown"

        try:
            # Convert local coordinates to BasicPoint2d
            point = BasicPoint2d(x, y)
            print(f"DEBUG (MapSensor): Created BasicPoint2d({x}, {y})") # DEBUG

            # Lanelet tutorial to: https://github.com/fzi-forschungszentrum-informatik/Lanelet2/blob/master/lanelet2_examples/scripts/tutorial.py
            # Find the nearest lanelet
            # Note: findNearest requires the point in the map's coordinate frame.
            # If the map was loaded with a projector, the map elements are already in that frame.
            # The track_data coordinates (xCenter, yCenter) are assumed to be in the same frame.
            nearest_lanelets = findNearest(lanelet_map.laneletLayer, point, 1)
            print(f"DEBUG (MapSensor): Found {len(nearest_lanelets)} nearest lanelets.") # DEBUG

            if nearest_lanelets:
                # nearest_lanelets is a list of tuples (distance, lanelet)
                nearest_lanelet = nearest_lanelets[0][1]
                print(f"DEBUG (MapSensor): Nearest lanelet ID: {nearest_lanelet.id}") # DEBUG

                if "subtype" in nearest_lanelet.attributes:
                    subtype = nearest_lanelet.attributes["subtype"]
                    print(f"DEBUG (MapSensor): Found subtype attribute: {subtype}") # DEBUG
                    if subtype in self.subtype_probs:
                        return subtype
                    else:
                        print(f"Warning (MapSensor): Unknown lanelet subtype '{subtype}' found.")
                        return "unknown"
                else:
                    print("DEBUG (MapSensor): No subtype attribute found, defaulting to 'road'.") # DEBUG
                    return "road" # Default guess if no subtype
            else:
                print("DEBUG (MapSensor): No nearest lanelets found, returning 'unknown'.") # DEBUG
                return "unknown"

        except Exception as e:
            print(f"ERROR (MapSensor): Error querying Lanelet2 map at ({x}, {y}): {e}")
            return "unknown"

    def generate_classification_bba(self, track_data: pd.Series,
                                   track_meta: pd.Series,
                                   **kwargs) -> MassFunction:
        """
        Generate a Basic Belief Assignment for classification based on map data.

        Args:
            track_data: Series containing track data for a specific frame
            track_meta: Series containing metadata for the track
            **kwargs: Must include 'lanelet_map' and 'projector'

        Returns:
            A mass function representing the map's belief about the object class
        """
        # Use GROUND TRUTH position for map context
        x_center = track_data["xCenter"]
        y_center = track_data["yCenter"]

        lanelet_map = kwargs.get("lanelet_map")
        projector = kwargs.get("projector")

        # # --- DEBUG ---
        # print(f"DEBUG (MapSensor): generate_classification_bba called for track {track_meta.name}")
        # print(f"DEBUG (MapSensor): Received lanelet_map is None = {lanelet_map is None}")
        # print(f"DEBUG (MapSensor): Received projector is None = {projector is None}")
        # # --- END DEBUG ---

        subtype = self._get_lanelet_subtype(x_center, y_center, lanelet_map, projector)
        region_probs = self.subtype_probs.get(subtype, self.subtype_probs["unknown"])

        mass_dict = {}
        scaling_factor = 0.8 # Base belief in map context
        for class_name, prob in region_probs.items():
            if class_name in self.frame and prob > 0:
                mass_dict[frozenset({class_name})] = prob * scaling_factor

        # Assign remaining mass to the universal set (uncertainty)
        total_assigned_mass = sum(mass_dict.values())
        if total_assigned_mass < 1.0:
             # Ensure universal set mass reflects the base uncertainty (1 - scaling_factor)
             # plus any unassigned probability mass from the subtype rules.
             mass_dict[self.frame] = max(1.0 - total_assigned_mass, 1.0 - scaling_factor)

        # Normalize if necessary (shouldn't be needed with the logic above, but good practice)
        current_total = sum(mass_dict.values())
        if not np.isclose(current_total, 1.0):
            if current_total > 1e-9:
                norm_factor = 1.0 / current_total
                for k in mass_dict:
                    mass_dict[k] *= norm_factor
            else: # Avoid division by zero if all probs were somehow zero
                mass_dict = {self.frame: 1.0}

        mass_function = MassFunction(self.frame, mass_dict)

        # Apply overall sensor reliability discounting
        if self.reliability < 1.0:
            mass_function = mass_function.discount(self.reliability)

        return mass_function

