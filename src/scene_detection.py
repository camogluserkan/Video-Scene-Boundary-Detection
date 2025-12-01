"""
Scene Detection Module

This module orchestrates Phase 2 of the project: grouping shots into semantic scenes.
It implements the "Optimal Sequential Grouping" (OSG) algorithm using the H_nrm cost function.

Process:
1. Extract features from shot keyframes (using CLIP).
2. Calculate a pairwise cosine distance matrix.
3. Estimate the optimal number of scenes (K).
4. Group shots into scenes using the H_nrm algorithm.
5. Map results back to timecodes and generate a report.
"""

import os
import sys
import json
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional

# Add the cloned algorithm repository to the Python path
# This allows us to import h_nrm and estimate_scenes_count
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
algo_path = os.path.join(project_root, 'cluster_dp_algo', 'video-scene-detection')
if algo_path not in sys.path:
    sys.path.append(algo_path)

try:
    from h_nrm import get_optimal_sequence_nrm
    from estimate_scenes_count import estimate_scenes_count
except ImportError:
    print(f"Warning: Could not import OSG algorithms from {algo_path}")
    print("Make sure the 'cluster_dp_algo' submodule is correctly cloned.")

from src.feature_extraction import FeatureExtractor
from src.utils.visualization import save_shot_keyframes
from src.utils.video_utils import get_video_info


class SceneDetector:
    """
    Main class for Scene Detection.
    """

    def __init__(self, feature_extractor: Optional[FeatureExtractor] = None):
        """
        Initialize the SceneDetector.

        Args:
            feature_extractor: An instance of FeatureExtractor. If None, creates a new one.
        """
        self.feature_extractor = feature_extractor if feature_extractor else FeatureExtractor()

    def detect_scenes(self, video_path: str, shot_boundaries: List[int], keyframes_dir: str) -> List[Dict]:
        """
        Execute the scene detection pipeline.

        Args:
            video_path: Path to the input video.
            shot_boundaries: List of frame indices where shots end (from Phase 1).
            keyframes_dir: Directory where keyframes for each shot are stored.

        Returns:
            List[Dict]: A list of detected scenes with metadata.
        """
        print(f"\nStarting Scene Detection for: {video_path}")
        
        # 1. Keyframe Verification
        # We need one image per shot.
        # shot_boundaries length = N-1 (if N shots). 
        # But usually the shot detector returns just the cuts.
        # Total shots = len(shot_boundaries) + 1.
        
        expected_shots = len(shot_boundaries) + 1
        
        # Check if keyframes exist
        existing_files = sorted([f for f in os.listdir(keyframes_dir) if f.endswith('.jpg')]) if os.path.exists(keyframes_dir) else []
        
        if len(existing_files) != expected_shots:
            print(f"Keyframes missing or count mismatch ({len(existing_files)} found, {expected_shots} expected).")
            print("Extracting keyframes from video...")
            save_shot_keyframes(video_path, shot_boundaries, keyframes_dir)
            existing_files = sorted([f for f in os.listdir(keyframes_dir) if f.endswith('.jpg')])
        else:
            print(f"✓ Found {len(existing_files)} existing keyframes.")

        keyframe_paths = [os.path.join(keyframes_dir, f) for f in existing_files]

        # 2. Feature Extraction
        print("Extracting semantic features with CLIP...")
        features = self.feature_extractor.extract_features(keyframe_paths)
        
        # Save features for caching/debugging
        features_path = os.path.join(os.path.dirname(keyframes_dir), "..", "features", 
                                   f"{os.path.basename(video_path).split('.')[0]}_features.npy")
        self.feature_extractor.save_features(features, features_path)

        # 3. Distance Matrix Construction
        # Cosine Distance = 1 - Cosine Similarity
        # Features are already normalized, so Cosine Similarity = A . A^T
        print("Calculating distance matrix...")
        similarity_matrix = np.dot(features, features.T)
        
        # Clip values to [0, 1] range to avoid numerical errors slightly outside this range
        similarity_matrix = np.clip(similarity_matrix, 0, 1)
        
        # Convert to distance
        distance_matrix = 1.0 - similarity_matrix
        
        # Ensure main diagonal is 0 (self-distance)
        np.fill_diagonal(distance_matrix, 0)

        # 4. Optimal Sequential Grouping (OSG)
        # 4.1 Estimate K (Number of scenes)
        print("Estimating optimal scene count (K)...")
        try:
            estimated_k = estimate_scenes_count(distance_matrix)
            # Ensure at least 1 scene and at most N scenes
            estimated_k = max(1, min(estimated_k, len(keyframe_paths)))
            print(f"Estimated Scene Count: {estimated_k}")
        except Exception as e:
            print(f"Error estimating K: {e}. Defaulting to N/5.")
            estimated_k = max(1, len(keyframe_paths) // 5)

        # 4.2 Group shots
        print(f"Grouping {len(keyframe_paths)} shots into {estimated_k} scenes using H_nrm...")
        try:
            # get_optimal_sequence_nrm returns indices of the LAST shot of each scene
            # The indices are 0-based relative to the shot list.
            scene_boundaries_indices = get_optimal_sequence_nrm(distance_matrix, estimated_k)
        except Exception as e:
            print(f"Error in OSG algorithm: {e}. Fallback to single scene.")
            scene_boundaries_indices = [len(keyframe_paths) - 1]

        # 5. Construct Output
        scenes = self._construct_scene_objects(video_path, shot_boundaries, scene_boundaries_indices, keyframe_paths)
        
        return scenes

    def _construct_scene_objects(self, video_path: str, 
                                shot_boundaries: List[int], 
                                scene_split_indices: List[int],
                                keyframe_paths: List[str]) -> List[Dict]:
        """
        Helper to format the raw algorithm output into a structured list of scenes.
        """
        video_info = get_video_info(video_path)
        fps = video_info['fps']
        
        # Reconstruct shot start/end frames
        shot_starts = [0] + shot_boundaries
        shot_ends = shot_boundaries + [video_info['frame_count']]
        
        scenes = []
        current_shot_idx = 0
        
        for i, split_idx in enumerate(scene_split_indices):
            # scene_split_indices contains the index of the LAST shot in the scene
            # e.g., if shots are 0, 1, 2, 3 and split is at 1, 3
            # Scene 1: shots 0, 1. Scene 2: shots 2, 3.
            
            last_shot_idx = int(split_idx)
            
            scene_shots_indices = list(range(current_shot_idx, last_shot_idx + 1))
            
            # Get time range from the first shot's start to the last shot's end
            scene_start_frame = shot_starts[scene_shots_indices[0]]
            scene_end_frame = shot_ends[scene_shots_indices[-1]]
            
            scene_obj = {
                "scene_id": i + 1,
                "start_frame": int(scene_start_frame),
                "end_frame": int(scene_end_frame),
                "start_time": self._format_time(scene_start_frame / fps),
                "end_time": self._format_time(scene_end_frame / fps),
                "shot_indices": scene_shots_indices, # 0-based indices of shots in this scene
                "keyframe": keyframe_paths[scene_shots_indices[len(scene_shots_indices)//2]] # Middle shot's keyframe
            }
            scenes.append(scene_obj)
            
            current_shot_idx = last_shot_idx + 1
            
        return scenes

    def _format_time(self, seconds: float) -> str:
        """Formats seconds into HH:MM:SS."""
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

    def save_results(self, scenes: List[Dict], output_path: str):
        """Save the scenes list to a JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scenes, f, indent=2, ensure_ascii=False)
        print(f"Scene detection results saved to: {output_path}")

