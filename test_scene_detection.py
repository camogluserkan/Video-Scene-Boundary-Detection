"""
Scene Detection Test Script

This script runs the full pipeline:
1. Shot Boundary Detection (using TransNet V2)
2. Scene Detection (using CLIP features + OSG algorithm)
"""

import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.shot_detection import ShotDetector
from src.scene_detection import SceneDetector

def run_pipeline(video_path: str, output_dir: str):
    print("="*60)
    print("VIDEO SCENE DETECTION PIPELINE")
    print("="*60)

    # 1. Phase 1: Shot Detection
    print("\n--- PHASE 1: SHOT DETECTION ---")
    shot_detector = ShotDetector(method='transnet')
    try:
        shot_boundaries = shot_detector.detect(video_path)
    except Exception as e:
        print(f"Error during shot detection: {e}")
        return

    # Save shot results (optional, but good practice)
    shot_output_dir = os.path.join(output_dir, "shots")
    shot_detector.save_results(shot_output_dir, video_path)
    
    # 2. Phase 2: Scene Detection
    print("\n--- PHASE 2: SCENE DETECTION ---")
    
    # Define where keyframes are (ShotDetector saves them in a specific subfolder)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    keyframes_dir = os.path.join(shot_output_dir, f"{video_name}_keyframes")
    
    scene_detector = SceneDetector()
    
    try:
        scenes = scene_detector.detect_scenes(video_path, shot_boundaries, keyframes_dir)
    except Exception as e:
        print(f"Error during scene detection: {e}")
        import traceback
        traceback.print_exc()
        return

    # Save scene results
    scene_output_path = os.path.join(output_dir, f"{video_name}_scenes.json")
    scene_detector.save_results(scenes, scene_output_path)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Detected Scenes: {len(scenes)}")
    print(f"Report: {scene_output_path}")
    print("="*60)
    
    # Print preview
    for scene in scenes[:5]:
        print(f"Scene {scene['scene_id']}: {scene['start_time']} - {scene['end_time']} ({len(scene['shot_indices'])} shots)")
    if len(scenes) > 5:
        print("...")

if __name__ == "__main__":
    # Default to sample video
    default_video = "data/videos/sample_video_v3.mp4"
    
    if not os.path.exists(default_video):
        print(f"Sample video not found at {default_video}. Please provide a video path.")
        sys.exit(1)
        
    run_pipeline(default_video, "data/results")
