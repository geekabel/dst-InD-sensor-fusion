"""
Main script for testing and evaluating the Dempster-Shafer sensor fusion system.

This script loads data from the inD dataset, simulates virtual sensors,
fuses their outputs using different combination rules, and evaluates
the classification performance.

Version 1.1: Updated to use Lanelet2 map data and realistic sensor noise models.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Set, Optional, Any

# Import modules from this package
from dst_core.basic import MassFunction
from data_loader.loader import (
    load_recording_meta, load_tracks_meta, load_tracks,
    get_track_data_by_frame, get_track_class
)
from sensors.virtual_sensors import (
    CameraSensor, RadarSensor, LidarSensor, MapSensor, LANELET2_AVAILABLE
)
from classification import (
    make_decision, calculate_accuracy, calculate_confusion_matrix,
    calculate_precision_recall_f1
)
from visualization.plotter import (
    plot_mass_function, plot_belief_plausibility
)

# Import Lanelet2 if available
try:
    import lanelet2
    from lanelet2.projection import UtmProjector
    from lanelet2.io import Origin, load
except ImportError:
    print("Warning: Lanelet2 library not found. MapSensor will use simplified logic.")


def run_experiment(
    dataset_path: str,
    recording_id: int,
    num_frames: int = 50,
    frame_step: int = 10,
    combination_rule: str = "dempster",
    decision_method: str = "max_pignistic",
    results_dir: str = "/home/godwin/Downloads/sensor_fusion_dst_package_v1.1/results",
    save_plots: bool = True,
    verbose: bool = True
) -> Tuple[float, Dict[str, float]]:
    """
    Run the sensor fusion experiment on a specific recording.
    
    Args:
        dataset_path: Path to the inD dataset directory
        recording_id: ID of the recording to process
        num_frames: Number of frames to process
        frame_step: Step size for frame processing (e.g., 10 means process every 10th frame)
        combination_rule: Which combination rule to use ('dempster', 'yager', or 'pcr5')
        decision_method: Decision method ('max_bel', 'max_pl', or 'max_pignistic')
        results_dir: Directory to save results
        save_plots: Whether to save plots
        verbose: Whether to print progress information
        
    Returns:
        Tuple of (accuracy, detailed_metrics_dict)
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Format recording ID as a string with leading zero if needed
    recording_id_str = f"{recording_id:02d}"
    
    # Load data
    if verbose:
        print(f"Loading data for recording {recording_id_str}...")
    
    recording_meta = load_recording_meta(
        os.path.join(dataset_path, "data"), recording_id
    )
    tracks_meta = load_tracks_meta(
        os.path.join(dataset_path, "data"), recording_id
    )
    tracks = load_tracks(
        os.path.join(dataset_path, "data"), recording_id
    )
    
    # Load Lanelet2 map if available
    lanelet_map = None
    projector = None
    if LANELET2_AVAILABLE:
        try:
            # Find the map file for this recording
            location_id = recording_meta.get("locationId", 1)
            map_file = os.path.join(dataset_path, "maps", f"location_{location_id}.osm")
            
            if os.path.exists(map_file):
                # Get UTM origin from recording metadata
                utm_origin_x = recording_meta.get("xUtmOrigin", 0)
                utm_origin_y = recording_meta.get("yUtmOrigin", 0)
                
                # Create projector with origin
                projector = UtmProjector(Origin(utm_origin_x, utm_origin_y))
                
                # Load the map
                lanelet_map = load(map_file, projector)
                
                if verbose:
                    print(f"Loaded Lanelet2 map from {map_file}")
            else:
                print(f"Warning: Map file {map_file} not found. MapSensor will use simplified logic.")
        except Exception as e:
            print(f"Error loading Lanelet2 map: {e}")
            print("MapSensor will use simplified logic.")
    
    # Define frame of discernment
    frame = {"car", "truck_bus", "bicycle", "pedestrian", "unknown"}
    
    # Create sensors with realistic noise models
    # Position sensors at the origin for simplicity
    camera = CameraSensor(frame, reliability=0.9, position=(0, 0), position_noise_std_dev=0.5)
    radar = RadarSensor(frame, reliability=0.8, position=(0, 0), 
                       range_noise_std_dev=0.3, lateral_noise_std_dev=1.0, velocity_noise_std_dev=0.2)
    lidar = LidarSensor(frame, reliability=0.85, position=(0, 0), position_noise_std_dev=0.2)
    map_sensor = MapSensor(frame, reliability=0.7)
    
    # Determine frames to process
    unique_frames = sorted(tracks["frame"].unique())
    frames_to_process = unique_frames[:num_frames*frame_step:frame_step]
    
    # Initialize results storage
    predictions = []
    ground_truths = []
    
    # Process each frame
    for i, frame_id in enumerate(frames_to_process):
        if verbose and (i % 10 == 0 or i == len(frames_to_process) - 1):
            print(f"Processing frame {frame_id} ({i+1}/{len(frames_to_process)})")
        
        # Get track IDs present in this frame
        track_ids = tracks[tracks["frame"] == frame_id]["trackId"].unique()
        
        # Process each track
        for track_id in track_ids:
            # Get track data for this frame
            track_data = get_track_data_by_frame(tracks, track_id, frame_id)
            if track_data is None:
                continue
                
            # Get track metadata using track_id as index
            try:
                track_meta = tracks_meta.loc[track_id]
            except KeyError:
                if verbose:
                    print(f"Warning: Track ID {track_id} not found in tracks_meta. Skipping.")
                continue
            
            # Generate BBAs from each sensor
            camera_bba = camera.generate_classification_bba(track_data, track_meta)
            radar_bba = radar.generate_classification_bba(track_data, track_meta)
            lidar_bba = lidar.generate_classification_bba(track_data, track_meta)
            map_bba = map_sensor.generate_classification_bba(
                track_data, track_meta, lanelet_map=lanelet_map, projector=projector
            )
            
            # Save example plots for the first few tracks in the first frame
            if save_plots and frame_id == frames_to_process[0] and track_id < 5:
                plot_mass_function(
                    camera_bba, 
                    title=f"Camera BBA - Track {track_id}",
                    save_path=os.path.join(results_dir, f"camera_bba_track{track_id}_frame{frame_id}.png")
                )
                plot_mass_function(
                    radar_bba, 
                    title=f"Radar BBA - Track {track_id}",
                    save_path=os.path.join(results_dir, f"radar_bba_track{track_id}_frame{frame_id}.png")
                )
                plot_mass_function(
                    lidar_bba, 
                    title=f"Lidar BBA - Track {track_id}",
                    save_path=os.path.join(results_dir, f"lidar_bba_track{track_id}_frame{frame_id}.png")
                )
                plot_mass_function(
                    map_bba, 
                    title=f"Map BBA - Track {track_id}",
                    save_path=os.path.join(results_dir, f"map_bba_track{track_id}_frame{frame_id}.png")
                )
            
            # Combine BBAs using the specified rule
            if combination_rule == "dempster":
                fused_bba = camera_bba.combine_dempster(radar_bba).combine_dempster(lidar_bba).combine_dempster(map_bba)
            elif combination_rule == "yager":
                fused_bba = camera_bba.combine_yager(radar_bba).combine_yager(lidar_bba).combine_yager(map_bba)
            elif combination_rule == "pcr5":
                fused_bba = camera_bba.combine_pcr5(radar_bba).combine_pcr5(lidar_bba).combine_pcr5(map_bba)
            else:
                raise ValueError(f"Unknown combination rule: {combination_rule}")
            
            # Save example plots for the first few tracks in the first frame
            if save_plots and frame_id == frames_to_process[0] and track_id < 5:
                plot_mass_function(
                    fused_bba, 
                    title=f"Fused BBA ({combination_rule}) - Track {track_id}",
                    save_path=os.path.join(results_dir, f"fused_bba_{combination_rule}_track{track_id}_frame{frame_id}.png")
                )
                plot_belief_plausibility(
                    fused_bba, 
                    title=f"Belief/Plausibility ({combination_rule}) - Track {track_id}",
                    save_path=os.path.join(results_dir, f"bel_pl_{combination_rule}_track{track_id}_frame{frame_id}.png")
                )
            
            # Make decision
            predicted_class = make_decision(fused_bba, method=decision_method)
            
            # Get ground truth
            true_class = track_meta["class"]
            
            # Store results
            predictions.append(predicted_class)
            ground_truths.append(true_class)
    
    # Calculate metrics
    all_labels = list(frame) # Get labels from the frame of discernment
    accuracy = calculate_accuracy(predictions, ground_truths)
    confusion_mat = calculate_confusion_matrix(predictions, ground_truths, labels=all_labels)
    metrics = calculate_precision_recall_f1(confusion_mat)
    
    # Save results
    pd.DataFrame({
        "predicted": predictions,
        "ground_truth": ground_truths
    }).to_csv(os.path.join(results_dir, f"predictions_{combination_rule}.csv"), index=False)
    
    pd.DataFrame(confusion_mat).to_csv(
        os.path.join(results_dir, f"confusion_matrix_{combination_rule}.csv")
    )
    
    pd.DataFrame(metrics).to_csv(
        os.path.join(results_dir, f"performance_metrics_{combination_rule}.csv")
    )
    
    if verbose:
        print(f"\nResults for {combination_rule} rule:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(pd.DataFrame(confusion_mat))
        print("\nPerformance Metrics:")
        print(pd.DataFrame(metrics))
    
    return accuracy, metrics


def compare_combination_rules(
    dataset_path: str,
    recording_id: int,
    num_frames: int = 50,
    frame_step: int = 10,
    decision_method: str = "max_pignistic",
    results_dir: str = "/home/godwin/results",
    save_plots: bool = True,
    verbose: bool = True
) -> None:
    """
    Compare different combination rules on the same dataset.
    
    Args:
        dataset_path: Path to the inD dataset directory
        recording_id: ID of the recording to process
        num_frames: Number of frames to process
        frame_step: Step size for frame processing
        decision_method: Decision method
        results_dir: Directory to save results
        save_plots: Whether to save plots
        verbose: Whether to print progress information
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Run experiments with different combination rules
    results = {}
    
    if verbose:
        print("=== Testing Dempster's combination rule ===")
    accuracy_dempster, metrics_dempster = run_experiment(
        dataset_path, recording_id, num_frames, frame_step,
        combination_rule="dempster", decision_method=decision_method,
        results_dir=results_dir, save_plots=save_plots, verbose=verbose
    )
    results["dempster"] = {
        "accuracy": accuracy_dempster,
        "precision_macro": metrics_dempster.loc["Macro Avg", "Precision"],
        "recall_macro": metrics_dempster.loc["Macro Avg", "Recall"],
        "f1_macro": metrics_dempster.loc["Macro Avg", "F1-Score"]
    }
    
    if verbose:
        print("\n=== Testing Yager's combination rule ===")
    accuracy_yager, metrics_yager = run_experiment(
        dataset_path, recording_id, num_frames, frame_step,
        combination_rule="yager", decision_method=decision_method,
        results_dir=results_dir, save_plots=save_plots, verbose=verbose
    )
    results["yager"] = {
        "accuracy": accuracy_yager,
        "precision_macro": metrics_yager.loc["Macro Avg", "Precision"],
        "recall_macro": metrics_yager.loc["Macro Avg", "Recall"],
        "f1_macro": metrics_yager.loc["Macro Avg", "F1-Score"]
    }
    
    if verbose:
        print("\n=== Testing PCR5 combination rule ===")
    accuracy_pcr5, metrics_pcr5 = run_experiment(
        dataset_path, recording_id, num_frames, frame_step,
        combination_rule="pcr5", decision_method=decision_method,
        results_dir=results_dir, save_plots=save_plots, verbose=verbose
    )
    results["pcr5"] = {
        "accuracy": accuracy_pcr5,
        "precision_macro": metrics_pcr5.loc["Macro Avg", "Precision"],
        "recall_macro": metrics_pcr5.loc["Macro Avg", "Recall"],
        "f1_macro": metrics_pcr5.loc["Macro Avg", "F1-Score"]
    }
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        "dempster": [
            results["dempster"]["accuracy"],
            results["dempster"]["precision_macro"],
            results["dempster"]["recall_macro"],
            results["dempster"]["f1_macro"]
        ],
        "yager": [
            results["yager"]["accuracy"],
            results["yager"]["precision_macro"],
            results["yager"]["recall_macro"],
            results["yager"]["f1_macro"]
        ],
        "pcr5": [
            results["pcr5"]["accuracy"],
            results["pcr5"]["precision_macro"],
            results["pcr5"]["recall_macro"],
            results["pcr5"]["f1_macro"]
        ]
    }, index=["Accuracy", "Precision (Macro Avg)", "Recall (Macro Avg)", "F1-Score (Macro Avg)"])
    
    # Save comparison table
    comparison_df.to_csv(os.path.join(results_dir, "combination_rules_comparison.csv"))
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    comparison_df.plot(kind="bar", rot=0)
    plt.title("Comparison of Combination Rules")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "combination_rules_comparison.png"))
    plt.close()
    
    if verbose:
        print("\n=== Comparison of Combination Rules ===")
        print(comparison_df)
        print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Dempster-Shafer sensor fusion experiment")
    parser.add_argument("--dataset_path", type=str, default="/home/godwin/Downloads/sensor_fusion_dst_package_v1.1/sensor_fusion_dst/inD-dataset-v1.1",
                       help="Path to the inD dataset directory")
    parser.add_argument("--recording_id", type=int, default=1,
                       help="ID of the recording to process")
    parser.add_argument("--num_frames", type=int, default=50,
                       help="Number of frames to process")
    parser.add_argument("--frame_step", type=int, default=10,
                       help="Step size for frame processing")
    parser.add_argument("--decision_method", type=str, default="max_pignistic",
                       choices=["max_bel", "max_pl", "max_pignistic"],
                       help="Decision method")
    parser.add_argument("--results_dir", type=str, default="/home/godwin/Downloads/sensor_fusion_dst_package_v1.1/results",
                       help="Directory to save results")
    parser.add_argument("--no_plots", action="store_true",
                       help="Disable saving plots")
    parser.add_argument("--quiet", action="store_true",
                       help="Disable verbose output")
    
    args = parser.parse_args()
    
    # Run comparison
    compare_combination_rules(
        dataset_path=args.dataset_path,
        recording_id=args.recording_id,
        num_frames=args.num_frames,
        frame_step=args.frame_step,
        decision_method=args.decision_method,
        results_dir=args.results_dir,
        save_plots=not args.no_plots,
        verbose=not args.quiet
    )
