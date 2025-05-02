"""
Module for loading and processing data from the inD dataset.

This module provides functions to read the recording metadata, track metadata,
and track time-series data from the CSV files provided in the inD dataset.
"""

import os
import pandas as pd
from typing import Dict, List, Tuple, Optional


def load_recording_meta(data_path: str, recording_id: int) -> Optional[pd.Series]:
    """
    Load metadata for a specific recording.

    Args:
        data_path: Path to the directory containing the dataset files (e.g., /home/ubuntu/inD_dataset/data).
        recording_id: The ID of the recording to load (e.g., 1 for 01_recordingMeta.csv).

    Returns:
        A pandas Series containing the metadata for the recording, or None if the file is not found.
    """
    meta_file = os.path.join(data_path, f"{recording_id:02d}_recordingMeta.csv")
    if not os.path.exists(meta_file):
        print(f"Warning: Recording metadata file not found: {meta_file}")
        return None
    
    try:
        # Read the CSV, assuming only one row of data after the header
        df = pd.read_csv(meta_file)
        if not df.empty:
            return df.iloc[0] # Return the first (and likely only) row as a Series
        else:
            print(f"Warning: Recording metadata file is empty: {meta_file}")
            return None
    except Exception as e:
        print(f"Error loading recording metadata from {meta_file}: {e}")
        return None

def load_tracks_meta(data_path: str, recording_id: int) -> Optional[pd.DataFrame]:
    """
    Load track metadata for a specific recording.

    Args:
        data_path: Path to the directory containing the dataset files.
        recording_id: The ID of the recording.

    Returns:
        A pandas DataFrame containing the track metadata, indexed by trackId, or None if the file is not found.
    """
    meta_file = os.path.join(data_path, f"{recording_id:02d}_tracksMeta.csv")
    if not os.path.exists(meta_file):
        print(f"Warning: Tracks metadata file not found: {meta_file}")
        return None
    
    try:
        df = pd.read_csv(meta_file)
        # Set trackId as index for easier lookup
        df.set_index("trackId", inplace=True)
        return df
    except Exception as e:
        print(f"Error loading tracks metadata from {meta_file}: {e}")
        return None

def load_tracks(data_path: str, recording_id: int) -> Optional[pd.DataFrame]:
    """
    Load the time-series track data for a specific recording.

    Args:
        data_path: Path to the directory containing the dataset files.
        recording_id: The ID of the recording.

    Returns:
        A pandas DataFrame containing the track data, or None if the file is not found.
    """
    tracks_file = os.path.join(data_path, f"{recording_id:02d}_tracks.csv")
    if not os.path.exists(tracks_file):
        print(f"Warning: Tracks file not found: {tracks_file}")
        return None
    
    try:
        df = pd.read_csv(tracks_file)
        return df
    except Exception as e:
        print(f"Error loading tracks data from {tracks_file}: {e}")
        return None

def get_track_data_by_frame(tracks_df: pd.DataFrame, track_id: int, frame: int) -> Optional[pd.Series]:
    """
    Retrieve the data for a specific track at a specific frame.

    Args:
        tracks_df: DataFrame containing the time-series track data (from load_tracks).
        track_id: The ID of the track.
        frame: The frame number.

    Returns:
        A pandas Series containing the track data for the specified frame, or None if not found.
    """
    track_frame_data = tracks_df[(tracks_df["trackId"] == track_id) & (tracks_df["frame"] == frame)]
    if not track_frame_data.empty:
        return track_frame_data.iloc[0]
    else:
        return None

def get_track_class(tracks_meta_df: pd.DataFrame, track_id: int) -> Optional[str]:
    """
    Get the ground truth class for a specific track.

    Args:
        tracks_meta_df: DataFrame containing track metadata (from load_tracks_meta).
        track_id: The ID of the track.

    Returns:
        The class label (str) or None if the trackId is not found.
    """
    try:
        return tracks_meta_df.loc[track_id, "class"]
    except KeyError:
        print(f"Warning: Track ID {track_id} not found in tracks metadata.")
        return None

# Example Usage (can be removed or placed under if __name__ == "__main__")
# if __name__ == "__main__":
#     dataset_path = "/home/ubuntu/inD_dataset/data"
#     rec_id = 1
# 
#     recording_info = load_recording_meta(dataset_path, rec_id)
#     if recording_info is not None:
#         print("Recording Meta:")
#         print(recording_info)
#         print("-"*20)
# 
#     tracks_meta_info = load_tracks_meta(dataset_path, rec_id)
#     if tracks_meta_info is not None:
#         print("Tracks Meta (first 5 rows):")
#         print(tracks_meta_info.head())
#         print("-"*20)
#         # Get class for track 2
#         track_2_class = get_track_class(tracks_meta_info, 2)
#         print(f"Class for Track 2: {track_2_class}")
#         print("-"*20)
# 
#     tracks_data = load_tracks(dataset_path, rec_id)
#     if tracks_data is not None:
#         print("Tracks Data (first 5 rows):")
#         print(tracks_data.head())
#         print("-"*20)
#         # Get data for track 0 at frame 10
#         track_0_frame_10 = get_track_data_by_frame(tracks_data, 0, 10)
#         if track_0_frame_10 is not None:
#             print("Data for Track 0, Frame 10:")
#             print(track_0_frame_10)
#         else:
#             print("Data for Track 0, Frame 10 not found.")

