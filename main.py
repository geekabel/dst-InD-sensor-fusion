"""
Main script for testing and running the sensor fusion classification task.
"""
import os
import argparse
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import sys
from dotenv import load_dotenv # Import load_dotenv
import configparser # Import configparser
from typing import Tuple, Dict, Any, Optional, List, Set, Union

# Load environment variables from .env file
load_dotenv()

# Try importing Lanelet2
LANELET2_AVAILABLE = False
try:
    import lanelet2
    from lanelet2.projection import LocalCartesianProjector
    from lanelet2.io import Origin
    LANELET2_AVAILABLE = True
    print("Lanelet2 library found and imported successfully.")
except ImportError:
    print("Warning: Lanelet2 library not found. MapSensor will use simplified logic.")
    # Define dummy classes if lanelet2 is not available to avoid NameErrors later
    class Origin:
        pass
    class LocalCartesianProjector:
        pass

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# Correctly import functions from data_loader
from data_loader.loader import load_recording_meta, load_tracks_meta, load_tracks
from sensors.virtual_sensors import CameraSensor, RadarSensor, LidarSensor, MapSensor
from dst_core.basic import MassFunction
# Import individual metric functions from classification
from classification import calculate_accuracy, calculate_confusion_matrix, calculate_precision_recall_f1
# Import correct visualization function names
from visualization.plotter import plot_mass_function, plot_belief_plausibility, plot_metrics_comparison

# Define the frame of discernment (possible classes)
FRAME_OF_DISCERNMENT = {"car", "van", "truck_bus", "pedestrian", "bicycle"}

# --- Configuration --- #
# CONFIG_FILE = os.path.join(PROJECT_ROOT, "config.ini")
DEFAULT_DATASET_PATH = "" # Default path if not in CLI or config
DEFAULT_RESULTS_PATH = "" # Default results path

def get_configured_path(cli_arg_path: Union[str,None], env_var_name: str, default_path: str) -> str:
    """Determine the dataset path based on CLI, config file, and default."""
    env_path = os.getenv(env_var_name)
    # We verify if CLI argument is provide by the user
    if cli_arg_path is not None:
        # For results_dir, argparse default is DEFAULT_RESULTS_PATH. If cli_arg_path is this default AND env_path exists, prefer .env
        if env_var_name == "RESULTS_PATH" and cli_arg_path == DEFAULT_DATASET_PATH and env_path:
            print(f"Using {env_var_name} from .env file: {env_path}")
            return env_path
        # For dataset_path (default=None in argparse) or if results_dir was explicitly set via CLI (not the default)
        if not (env_var_name == "RESULTS_PATH" and cli_arg_path == DEFAULT_RESULTS_PATH):
            print(f"Using {env_var_name} from command line: {cli_arg_path}")
            return cli_arg_path

    if env_path:
        print(f"Using {env_var_name} from .env file: {env_path}")
        return env_path

    print(f"Using default {env_var_name}: {default_path}")
    return default_path
# Define helper functions for combining BBAs using different rules
def combine_bba_list_dempster(bba_list):
    """Combine a list of BBAs using Dempster's rule."""
    if not bba_list:
        return None
    result = bba_list[0]
    for i in range(1, len(bba_list)):
        try:
            result = result.combine_dempster(bba_list[i])
        except ValueError as e:
            # Handle complete conflict by returning None or a vacuous BBA
            print(f"Warning: Complete conflict detected during Dempster combination: {e}")
            # Option 1: Return None (indicates failure to combine)
            # return None
            # Option 2: Return a vacuous BBA (total uncertainty)
            return MassFunction(result.frame, {result.frame: 1.0})
    return result

def combine_bba_list_yager(bba_list):
    """Combine a list of BBAs using Yager's rule."""
    if not bba_list:
        return None
    result = bba_list[0]
    for i in range(1, len(bba_list)):
        result = result.combine_yager(bba_list[i])
    return result

def combine_bba_list_pcr5(bba_list):
    """Combine a list of BBAs using PCR5 rule."""
    if not bba_list:
        return None
    result = bba_list[0]
    for i in range(1, len(bba_list)):
        result = result.combine_pcr5(bba_list[i])
    return result

def bba_to_decision(bba):
    """Convert a BBA to a decision using pignistic transformation."""
    if bba is None: # Handle case where combination failed
        return None, 0.0
    pignistic = bba.pignistic_transformation()
    if not pignistic:
        return None, 0.0
    # Find the class with highest pignistic probability
    best_class = max(pignistic.items(), key=lambda x: x[1])
    return best_class[0], best_class[1]  # Return (class, confidence)

def main(args):
    """Main function to run the sensor fusion experiment."""
    start_time = time.time()

    # --- Determine Dataset Path --- #
    dataset_path = get_configured_path(args.dataset_path, "DATASET_PATH", DEFAULT_DATASET_PATH)
    results_dir = get_configured_path(args.results_dir, "RESULTS_PATH", DEFAULT_RESULTS_PATH)

    # --- 1. Load Data --- #
    if args.verbose:
        print(f"Loading data for recording {args.recording_id} from {dataset_path}...")

    if not os.path.exists(results_dir):
        try: os.makedirs(results_dir)
        except OSError as e: print(f"Error; Could not create rresult directory {results_dir}: {e}"); return
    try:
        # Construct data path
        data_dir = os.path.join(dataset_path, "data")

        # Load individual data components
        recording_meta = load_recording_meta(data_dir, args.recording_id)
        tracks_meta = load_tracks_meta(data_dir, args.recording_id)
        tracks = load_tracks(data_dir, args.recording_id)

        if tracks is None or tracks_meta is None or recording_meta is None:
            print(f"Error: Could not load all required data components for recording {args.recording_id}. Exiting.")
            return

        location_id = recording_meta["locationId"] # Access directly from Series
        if args.verbose:
            print(f"Data loaded successfully. Location ID: {location_id}")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 2. Load Lanelet2 Map (if available) --- #
    lanelet_map = None
    projector = None
    if LANELET2_AVAILABLE:
        # Construct map file path based on location ID
        location_name_map = {
            1: "01_bendplatz",
            2: "02_frankenburg",
            3: "03_heckstrasse",
            4: "04_aseag"
        }
        location_folder_name = location_name_map.get(location_id)

        if location_folder_name:
            map_filename = f"location{location_id}.osm"
            # Use the determined dataset_path here
            map_file = os.path.join(dataset_path, "maps", "lanelets", location_folder_name, map_filename)
            if args.verbose:
                print(f"Attempting to load Lanelet2 map: {map_file}")

            if os.path.exists(map_file):
                try:
                    origin_lat = recording_meta["latLocation"]
                    origin_lon = recording_meta["lonLocation"]
                    if args.verbose:
                        print(f"Using LocalCartesianProjector with Origin(lat={origin_lat}, lon={origin_lon})")

                    origin = Origin(origin_lat, origin_lon)
                    projector = LocalCartesianProjector(origin)

                    lanelet_map, load_errors = lanelet2.io.loadRobust(map_file, projector)
                    if load_errors:
                        print("Warning: Errors encountered during Lanelet2 map loading:")
                        for error in load_errors:
                            print(f"- {error}")
                        lanelet_map = None
                        projector = None
                    else:
                        if args.verbose:
                            print("Lanelet2 map loaded successfully.")

                except Exception as e:
                    print(f"Error loading Lanelet2 map: {e}. MapSensor will use simplified logic.")
                    lanelet_map = None
                    projector = None
            else:
                print(f"Warning: Lanelet2 map file not found: {map_file}. MapSensor will use simplified logic.")
        else:
            print(f"Warning: Unknown location ID {location_id} for Lanelet2 map path. MapSensor will use simplified logic.")
    else:
        if args.verbose:
            print("Lanelet2 library not available. Skipping map loading.")

    # --- 3. Initialize Sensors --- #
    # Updated parameter names to match the class definitions
    camera = CameraSensor(frame_of_discernment=FRAME_OF_DISCERNMENT, position_noise_std_dev=0.5)
    radar = RadarSensor(frame_of_discernment=FRAME_OF_DISCERNMENT, range_noise_std_dev=0.2, lateral_noise_std_dev=1.0, velocity_noise_std_dev=0.3)
    lidar = LidarSensor(frame_of_discernment=FRAME_OF_DISCERNMENT, position_noise_std_dev=0.3)
    map_sensor = MapSensor(frame_of_discernment=FRAME_OF_DISCERNMENT, reliability=0.7)

    sensors = [camera, radar, lidar, map_sensor]

    # --- 4. Process Frames --- #
    results = {
        "dempster": {"predictions": [], "true_labels": [], "confidences": []},
        "yager": {"predictions": [], "true_labels": [], "confidences": []},
        "pcr5": {"predictions": [], "true_labels": [], "confidences": []}
    }
    processed_frames = 0

    start_frame = tracks["frame"].min()
    end_frame = tracks["frame"].max()
    target_frames = list(range(start_frame, min(end_frame + 1, start_frame + args.num_frames * args.frame_step), args.frame_step))

    if args.verbose:
        print(f"Processing {len(target_frames)} frames (from {start_frame} to {target_frames[-1]} with step {args.frame_step})...")

    frame_iterator = tqdm(target_frames, desc="Processing Frames", disable=args.quiet)

    for frame_num in frame_iterator:
        # Get all data for the current frame
        frame_tracks_data = tracks[tracks["frame"] == frame_num]
        if frame_tracks_data.empty:
            continue

        processed_tracks_in_frame = 0
        for track_id in frame_tracks_data["trackId"].unique():
            # Get the specific point for this track in this frame
            track_point = frame_tracks_data[frame_tracks_data["trackId"] == track_id].iloc[0]

            try:
                track_meta = tracks_meta.loc[track_id]
            except KeyError:
                 if args.verbose > 1:
                    print(f"Frame {frame_num}, Track {track_id}: Metadata not found using .loc, skipping.")
                 continue

            true_label = track_meta["class"]

            if true_label not in FRAME_OF_DISCERNMENT:
                if args.verbose > 1:
                    print(f"Frame {frame_num}, Track {track_id}: True label ", true_label, " not in Frame of Discernment, skipping.")
                continue

            bba_list = []
            for sensor in sensors:
                sensor_kwargs = {}
                if isinstance(sensor, MapSensor):
                    sensor_kwargs["lanelet_map"] = lanelet_map
                    sensor_kwargs["projector"] = projector
                sensor_kwargs["track_meta"] = track_meta

                # Call generate_classification_bba with track_point and kwargs
                bba = sensor.generate_classification_bba(track_point, **sensor_kwargs)
                if bba:
                    bba_list.append(bba)

            if not bba_list:
                if args.verbose > 1:
                    print(f"Frame {frame_num}, Track {track_id}: No BBAs generated, skipping.")
                continue

            try:
                fused_bba_dempster = combine_bba_list_dempster(bba_list)
                fused_bba_yager = combine_bba_list_yager(bba_list)
                fused_bba_pcr5 = combine_bba_list_pcr5(bba_list)
            except Exception as e:
                print(f"\nError during BBA combination for Frame {frame_num}, Track {track_id}: {e}")
                if args.verbose > 1:
                    print("BBAs:", bba_list)
                continue

            pred_dempster, conf_dempster = bba_to_decision(fused_bba_dempster)
            pred_yager, conf_yager = bba_to_decision(fused_bba_yager)
            pred_pcr5, conf_pcr5 = bba_to_decision(fused_bba_pcr5)

            # Only record results if a decision was made
            if pred_dempster is not None:
                results["dempster"]["predictions"].append(pred_dempster)
                results["dempster"]["true_labels"].append(true_label)
                results["dempster"]["confidences"].append(conf_dempster)

            if pred_yager is not None:
                results["yager"]["predictions"].append(pred_yager)
                results["yager"]["true_labels"].append(true_label)
                results["yager"]["confidences"].append(conf_yager)

            if pred_pcr5 is not None:
                results["pcr5"]["predictions"].append(pred_pcr5)
                results["pcr5"]["true_labels"].append(true_label)
                results["pcr5"]["confidences"].append(conf_pcr5)

            processed_tracks_in_frame += 1

            if processed_frames == 0 and processed_tracks_in_frame == 1 and args.plot:
                if not os.path.exists(args.results_dir):
                    os.makedirs(args.results_dir)
                if args.verbose:
                    print(f"Plotting results for Frame {frame_num}, Track {track_id}...")
                for i, bba in enumerate(bba_list):
                    sensor_name = sensors[i].__class__.__name__
                    # Use correct function name: plot_mass_function
                    plot_mass_function(bba, f"{sensor_name} BBA (F{frame_num}, T{track_id})", os.path.join(args.results_dir, f"sensor_{sensor_name}_f{frame_num}_t{track_id}.png"))
                if fused_bba_dempster:
                    # Use correct function name: plot_mass_function
                    plot_mass_function(fused_bba_dempster, f"Fused BBA (Dempster) (F{frame_num}, T{track_id})", os.path.join(args.results_dir, f"fused_bba_dempster_f{frame_num}_t{track_id}.png"))
                    # Use correct function name: plot_belief_plausibility
                    plot_belief_plausibility(fused_bba_dempster, title=f"Bel/Pl (Dempster) (F{frame_num}, T{track_id})", save_path=os.path.join(args.results_dir, f"bel_pl_dempster_f{frame_num}_t{track_id}.png"))
                if fused_bba_yager:
                    # Use correct function name: plot_mass_function
                    plot_mass_function(fused_bba_yager, f"Fused BBA (Yager) (F{frame_num}, T{track_id})", os.path.join(args.results_dir, f"fused_bba_yager_f{frame_num}_t{track_id}.png"))
                    # Use correct function name: plot_belief_plausibility
                    plot_belief_plausibility(fused_bba_yager, title=f"Bel/Pl (Yager) (F{frame_num}, T{track_id})", save_path=os.path.join(args.results_dir, f"bel_pl_yager_f{frame_num}_t{track_id}.png"))
                if fused_bba_pcr5:
                    # Use correct function name: plot_mass_function
                    plot_mass_function(fused_bba_pcr5, f"Fused BBA (PCR5) (F{frame_num}, T{track_id})", os.path.join(args.results_dir, f"fused_bba_pcr5_f{frame_num}_t{track_id}.png"))
                    # Use correct function name: plot_belief_plausibility
                    plot_belief_plausibility(fused_bba_pcr5, title=f"Bel/Pl (PCR5) (F{frame_num}, T{track_id})", save_path=os.path.join(args.results_dir, f"bel_pl_pcr5_f{frame_num}_t{track_id}.png"))

        if processed_tracks_in_frame > 0:
            processed_frames += 1

    if not results["dempster"]["true_labels"]:
        print("\nNo tracks processed or no valid labels found. Cannot calculate metrics.")
        return

    all_metrics = {}
    labels = sorted(list(FRAME_OF_DISCERNMENT))
    print("\n--- Evaluation Metrics ---")
    for rule in results:
        print(f"\nCombination Rule: {rule.capitalize()}")
        true_labels = results[rule]["true_labels"]
        predictions = results[rule]["predictions"]

        if not true_labels or not predictions:
            print("No predictions made for this rule, skipping metrics.")
            continue

        # Calculate individual metrics using imported functions
        accuracy = calculate_accuracy(predictions, true_labels)
        conf_matrix = calculate_confusion_matrix(predictions, true_labels, labels)
        perf_metrics = calculate_precision_recall_f1(conf_matrix)

        # Store metrics for comparison plot
        all_metrics[rule] = {
            'accuracy': accuracy,
            'precision_macro': perf_metrics.loc['Macro Avg', 'Precision'],
            'recall_macro': perf_metrics.loc['Macro Avg', 'Recall'],
            'f1_macro': perf_metrics.loc['Macro Avg', 'F1-Score'],
            'report': perf_metrics.to_string() # Store the DataFrame as string for verbose output
        }

        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision (Macro Avg): {perf_metrics.loc['Macro Avg', 'Precision']:.3f}")
        print(f"Recall (Macro Avg): {perf_metrics.loc['Macro Avg', 'Recall']:.3f}")
        print(f"F1-Score (Macro Avg): {perf_metrics.loc['Macro Avg', 'F1-Score']:.3f}")
        if args.verbose:
            print("Classification Report (Per Class):")
            print(perf_metrics.drop('Macro Avg').to_string())
            print("\n Confusion Matrix:")
            print(conf_matrix.to_string())

    if args.plot and all_metrics:
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        plot_metrics_comparison(all_metrics, os.path.join(args.results_dir, "combination_rules_comparison.png"))

    total_time =  time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sensor fusion classification experiments using Dempster-Shafer Theory.")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help=f"Path to the root directory of the inD dataset. Overrides .env. (Default via .env or: {DEFAULT_DATASET_PATH})")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_PATH,
                        help=f"Directory to save output plots and CSV files. Overrides .env. (Default via .env or: {DEFAULT_RESULTS_PATH})")
    parser.add_argument("--recording_id", type=int, default=1, help="The ID of the recording to process (e.g., 1, 2, ... default: 1).")
    parser.add_argument("--num_frames", type=int, default=50, help="Number of frames to process from the start (default: 50).")
    parser.add_argument("--frame_step", type=int, default=10, help="Step size between processed frames (default: 10).")
    parser.add_argument("--plot", action="store_true", help="Generate and save plots.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase output verbosity (e.g., -v, -vv).")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress bar and verbose output (overrides -v).")
    args = parser.parse_args()

    main(args)

