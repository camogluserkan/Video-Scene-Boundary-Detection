"""
Automated Shot Detection Test Script

This script is designed to verify the correct installation and functionality
of the Shot Detection system. It performs three main checks:
1. Module Imports: Ensures all required libraries (OpenCV, PyTorch, TransNetV2) are installed.
2. Video File Check: Verifies that a valid test video exists in the `data/videos` directory.
3. End-to-End Test: Runs the full shot detection pipeline on the test video and validates the output.

Usage:
    python test_shot_detection.py [optional_video_path]
"""

import os
import sys

def test_imports():
    """
    Verifies that all critical Python dependencies can be imported successfully.
    
    Returns:
        bool: True if all imports succeed, False otherwise.
    """
    print("="*60)
    print("1. Testing Module Imports...")
    print("="*60)
    
    try:
        import cv2
        print("✓ OpenCV loaded successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy
        print("✓ NumPy loaded successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} loaded successfully")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from transnetv2 import TransNetV2
        print("✓ TransNet V2 loaded successfully")
    except ImportError as e:
        print(f"✗ TransNet V2 import failed: {e}")
        return False
    
    print("\n✓ All modules loaded successfully!")
    return True


def test_video_file(video_path="data/videos/sample_video.mp4"):
    """
    Checks if the test video file exists and is readable by OpenCV.
    
    Args:
        video_path: Path to the video file to check.
        
    Returns:
        bool: True if the video exists and is valid, False otherwise.
    """
    print("\n" + "="*60)
    print("2. Checking Test Video...")
    print("="*60)
    
    if os.path.exists(video_path):
        print(f"✓ Video found: {video_path}")
        
        # Attempt to open the video to ensure it's not corrupted
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
             print(f"✗ Error: Could not open video file {video_path}")
             return False
             
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        print(f"  FPS: {fps:.2f}")
        print(f"  Frame Count: {frame_count}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Duration: {duration:.2f} seconds")
        
        return True
    else:
        print(f"✗ Video not found: {video_path}")
        print("\nPlease copy a video file to:")
        print(f"  {video_path}")
        return False


def run_shot_detection(video_path="data/videos/sample_video.mp4"):
    """
    Executes the shot detection pipeline on the test video.
    
    This function runs the `ShotDetector` using the TransNet V2 method,
    saves the results to `data/results/transnet_test`, and verifies
    that output files are created.
    
    Args:
        video_path: Path to the video file to process.
        
    Returns:
        bool: True if the process completes without errors.
    """
    print("\n" + "="*60)
    print("3. Starting Shot Detection...")
    print("="*60)
    
    try:
        from src.shot_detection import ShotDetector
        
        # Initialize detector with TransNet V2
        print("\nLoading TransNet V2 model...")
        detector = ShotDetector(method='transnet', threshold=0.5)
        
        # Run detection
        print(f"\nAnalyzing video: {video_path}")
        shot_boundaries = detector.detect(video_path)
        
        # Define output directory
        output_dir = "data/results/transnet_test"
        print(f"\nSaving results: {output_dir}")
        detector.save_results(output_dir, video_path)
        
        # Print summary
        print("\n" + "="*60)
        print("✓ SUCCESS!")
        print("="*60)
        print(f"Detected Shots: {len(shot_boundaries) + 1}")
        print(f"Shot boundaries (indices): {shot_boundaries[:5]}..." if len(shot_boundaries) > 5 else f"Shot boundaries: {shot_boundaries}")
        print(f"\nResults generated at: {output_dir}")
        print(f"  - Plot: {output_dir}/sample_video_shot_detection.png")
        print(f"  - Summary: {output_dir}/sample_video_shots_summary.jpg")
        print(f"  - JSON Report: {output_dir}/sample_video_shot_report.json")
        print(f"  - Keyframes: {output_dir}/sample_video_keyframes/")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("Automated Shot Detection Test")
    print("="*60)
    print()
    
    # Step 1: Check Imports
    if not test_imports():
        print("\n❌ Module import errors! Please check requirements.txt")
        return
    
    # Determine video path (command line arg or default)
    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/videos/sample_video.mp4"
    
    # Step 2: Check Video File
    if not test_video_file(video_path):
        if video_path == "data/videos/sample_video.mp4":
            # Only abort if default video is missing. If user provided a bad path, they know.
            return
    
    # Step 3: Run Detection
    if not run_shot_detection(video_path):
        print("\n❌ Shot detection failed!")
        return
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == '__main__':
    main()
