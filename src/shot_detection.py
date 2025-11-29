"""
Shot Boundary Detection Module

This module serves as the main entry point for identifying shot boundaries in videos.
It orchestrates the detection process by providing a unified interface `ShotDetector`
that can utilize various underlying algorithms.

Supported Detection Methods:
1. TransNet V2 (Recommended):
   - A deep learning-based approach using 3D CNNs.
   - Provides the highest accuracy, capable of detecting both hard cuts and gradual transitions.
   - Requires the `transnetv2` library.

2. Histogram-based Detection:
   - Compares color histograms between consecutive frames.
   - Fast and effective for hard cuts where the color distribution changes significantly.
   - Less effective for gradual transitions or scenes with significant motion but similar colors.

3. Frame Difference Detection:
   - Calculates the pixel-wise absolute difference between consecutive frames.
   - Very fast but sensitive to object motion and camera movement.
   - Best used for simple videos with static backgrounds.

4. Adaptive Threshold Detection:
   - Uses a sliding window to calculate a dynamic threshold based on local frame difference statistics.
   - More robust than fixed threshold methods as it adapts to the video's pacing.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
import os
import json
import argparse

from src.utils.video_utils import (
    get_video_info,
    calculate_histogram,
    frame_difference,
    histogram_difference
)
from src.utils.visualization import (
    plot_shot_boundaries,
    create_shot_summary,
    save_shot_keyframes
)

# Attempt to import TransNet V2 detector. 
# If not installed, the functionality will be disabled, but other methods will work.
try:
    from src.transnet_detector import TransNetDetector
    TRANSNET_AVAILABLE = True
except ImportError:
    TRANSNET_AVAILABLE = False


class ShotDetector:
    """
    Main class for Shot Boundary Detection.
    
    This class initializes the selected detection method, processes the video,
    and stores the results including shot boundaries and frame difference metrics.
    """
    
    def __init__(self, method: str = 'transnet', threshold: Optional[float] = None):
        """
        Initialize the ShotDetector.

        Args:
            method: The detection algorithm to use. Options:
                    - 'transnet': Deep learning based (requires transnetv2).
                    - 'histogram': Color histogram comparison.
                    - 'frame_diff': Pixel-wise frame difference.
                    - 'adaptive': Adaptive thresholding on histogram differences.
            threshold: A manual threshold value for declaring a shot boundary.
                       If None, a default or automatic threshold will be calculated based on the method.
        """
        self.method = method
        # Set default threshold if not provided. TransNet works well with 0.5.
        # Other methods will calculate a statistical threshold later.
        self.threshold = threshold if threshold is not None else (0.5 if method == 'transnet' else None)
        
        self.video_info = None      # Will store FPS, duration, etc.
        self.differences = None     # Will store the calculated difference signal per frame
        self.shot_boundaries = []   # Will store the frame indices of detected boundaries
        self.transnet_detector = None
        
    def detect(self, video_path: str, 
               resize: Optional[Tuple[int, int]] = None) -> List[int]:
        """
        Execute the shot detection process on the specified video.
        
        Args:
            video_path: Full path to the input video file.
            resize: Optional tuple (width, height) to resize frames before processing.
                    Resizing (e.g., to 320x180) significantly speeds up histogram 
                    and frame difference methods without much loss in accuracy.
            
        Returns:
            List[int]: A list of frame indices where a shot change was detected.
        """
        print(f"Retrieving video info: {video_path}")
        self.video_info = get_video_info(video_path)
        print(f"FPS: {self.video_info['fps']:.2f}, "
              f"Frame Count: {self.video_info['frame_count']}, "
              f"Duration: {self.video_info['duration']:.2f}s")
        
        print(f"\nStarting shot detection using {self.method.upper()} method...")
        
        # Dispatch to the appropriate internal detection method
        if self.method == 'transnet':
            self.differences, self.shot_boundaries = self._detect_transnet(video_path)
        elif self.method == 'histogram':
            self.differences, self.shot_boundaries = self._detect_histogram(
                video_path, resize)
        elif self.method == 'frame_diff':
            self.differences, self.shot_boundaries = self._detect_frame_diff(
                video_path, resize)
        elif self.method == 'adaptive':
            self.differences, self.shot_boundaries = self._detect_adaptive(
                video_path, resize)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        print(f"\n✓ {len(self.shot_boundaries)} shot boundaries detected!")
        return self.shot_boundaries
    
    def _detect_transnet(self, video_path: str) -> Tuple[np.ndarray, List[int]]:
        """
        Implementation of TransNet V2 detection.
        
        TransNet V2 processes the video using a specialized 3D CNN.
        It handles its own frame extraction and processing efficiently.
        """
        if not TRANSNET_AVAILABLE:
            raise ImportError(
                "TransNet V2 is not available!\n"
                "To install: pip install transnetv2"
            )
        
        # Initialize the wrapper class for TransNet
        self.transnet_detector = TransNetDetector(threshold=self.threshold)
        
        # Perform detection. Returns indices and raw prediction scores.
        shot_boundaries, predictions = self.transnet_detector.detect(video_path)
        
        # Store predictions as 'differences' so we can visualize the confidence scores later.
        differences = predictions
        
        return differences, shot_boundaries
    
    def _detect_histogram(self, video_path: str, 
                         resize: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, List[int]]:
        """
        Implementation of Histogram-based detection.
        
        Iterates through the video frame by frame, calculates the color histogram
        for each frame, and computes the Chi-Square distance between consecutive histograms.
        """
        cap = cv2.VideoCapture(video_path)
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        differences = []
        
        # Read the first frame to initialize
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Could not read video!")
        
        if resize:
            prev_frame = cv2.resize(prev_frame, resize)
        
        prev_hist = calculate_histogram(prev_frame)
        
        # Use tqdm for a progress bar
        pbar = tqdm(total=frame_count-1, desc="Processing frames")
        
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            if resize:
                curr_frame = cv2.resize(curr_frame, resize)
            
            # Calculate histogram for current frame
            curr_hist = calculate_histogram(curr_frame)
            
            # Calculate difference metric between previous and current histogram
            diff = histogram_difference(prev_hist, curr_hist, method='chi-square')
            differences.append(diff)
            
            # Update state for next iteration
            prev_hist = curr_hist
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        differences = np.array(differences)
        
        # If no manual threshold is set, calculate one statistically.
        # Mean + 2 * Standard Deviation is a common heuristic for detecting outliers (cuts).
        if self.threshold is None:
            self.threshold = np.mean(differences) + 2 * np.std(differences)
            print(f"Automatic threshold calculated: {self.threshold:.4f}")
        
        # Identify frames where the difference exceeds the threshold
        shot_boundaries = self._find_boundaries(differences, self.threshold)
        
        return differences, shot_boundaries
    
    def _detect_frame_diff(self, video_path: str,
                          resize: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, List[int]]:
        """
        Implementation of Pixel-wise Frame Difference detection.
        
        Simple and fast: computes the average absolute pixel difference between consecutive frames.
        """
        cap = cv2.VideoCapture(video_path)
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        differences = []
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Could not read video!")
        
        if resize:
            prev_frame = cv2.resize(prev_frame, resize)
        
        pbar = tqdm(total=frame_count-1, desc="Processing frames")
        
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            if resize:
                curr_frame = cv2.resize(curr_frame, resize)
            
            # Calculate raw pixel difference
            diff = frame_difference(prev_frame, curr_frame)
            differences.append(diff)
            
            prev_frame = curr_frame
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        differences = np.array(differences)
        
        # Frame difference is noisy, so we use a higher statistical threshold (Mean + 3*StdDev)
        if self.threshold is None:
            self.threshold = np.mean(differences) + 3 * np.std(differences)
            print(f"Automatic threshold calculated: {self.threshold:.4f}")
        
        shot_boundaries = self._find_boundaries(differences, self.threshold)
        
        return differences, shot_boundaries
    
    def _detect_adaptive(self, video_path: str,
                        resize: Optional[Tuple[int, int]]) -> Tuple[np.ndarray, List[int]]:
        """
        Implementation of Adaptive Threshold detection.
        
        Instead of a single global threshold, this method uses a sliding window
        to calculate a local threshold. This helps handles videos with varying
        pacing or noise levels.
        """
        # We reuse the histogram difference metric as the base signal
        differences, _ = self._detect_histogram(video_path, resize)
        
        window_size = 50  # Look at 25 frames before and 25 after
        shot_boundaries = []
        
        print("Calculating adaptive threshold...")
        for i in tqdm(range(len(differences))):
            # Determine the window range around the current frame
            start = max(0, i - window_size // 2)
            end = min(len(differences), i + window_size // 2)
            
            # Calculate statistics for the local window
            window = differences[start:end]
            adaptive_threshold = np.mean(window) + 2 * np.std(window)
            
            # Check if current frame's difference spikes above the local context
            if differences[i] > adaptive_threshold:
                # Prevent detecting adjacent frames as separate cuts (debounce)
                if not shot_boundaries or (i - shot_boundaries[-1] > 10):
                    shot_boundaries.append(i)
        
        # Store the global mean threshold just for visualization reference
        self.threshold = np.mean(differences) + 2 * np.std(differences)
        print(f"Mean threshold (reference): {self.threshold:.4f}")
        
        return differences, shot_boundaries
    
    def _find_boundaries(self, differences: np.ndarray, 
                        threshold: float,
                        min_shot_length: int = 10) -> List[int]:
        """
        Helper function to identify boundary indices from a signal array.
        
        Args:
            differences: Array of difference scores per frame.
            threshold: The value above which a frame is considered a boundary.
            min_shot_length: Minimum number of frames allowed between two shots 
                             (prevents flickering/multiple detections for one cut).
            
        Returns:
            List[int]: Indices of confirmed boundaries.
        """
        boundaries = []
        
        for i, diff in enumerate(differences):
            if diff > threshold:
                # Apply debouncing: ignore if too close to the last detected boundary
                if not boundaries or (i - boundaries[-1] > min_shot_length):
                    boundaries.append(i)
        
        return boundaries
    
    def save_results(self, output_dir: str, video_path: str):
        """
        Save all detection results to the specified directory.
        
        Outputs:
        1. A plot showing the difference signal and detected cuts.
        2. A summary image containing keyframes from each shot.
        3. Individual keyframe images for every shot.
        4. A JSON report with precise timestamps and metadata.
        
        Args:
            output_dir: Target directory for output files.
            video_path: Original video path (used for naming output files).
        """
        os.makedirs(output_dir, exist_ok=True)
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 1. Generate and save the signal plot
        print("\n1. Creating plot...")
        plot_path = os.path.join(output_dir, f"{video_name}_shot_detection.png")
        plot_shot_boundaries(self.differences, self.shot_boundaries, 
                           self.threshold, save_path=plot_path)
        
        # 2. Generate a visual summary grid
        print("\n2. Creating shot summary...")
        summary_path = os.path.join(output_dir, f"{video_name}_shots_summary.jpg")
        create_shot_summary(video_path, self.shot_boundaries, summary_path)
        
        # 3. Save individual keyframes (useful for Phase 2: Clustering)
        print("\n3. Saving keyframes...")
        keyframes_dir = os.path.join(output_dir, f"{video_name}_keyframes")
        save_shot_keyframes(video_path, self.shot_boundaries, keyframes_dir)
        
        # 4. Save detailed JSON report
        print("\n4. Creating JSON report...")
        report = {
            'video_info': self.video_info,
            'method': self.method,
            'threshold': float(self.threshold),
            'total_shots': len(self.shot_boundaries) + 1,
            'shot_boundaries': [int(b) for b in self.shot_boundaries],
            'shot_durations': self._calculate_shot_durations()
        }
        
        json_path = os.path.join(output_dir, f"{video_name}_shot_report.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ All results saved to: {output_dir}")
    
    def _calculate_shot_durations(self) -> List[dict]:
        """
        Helper to calculate start/end times and duration for each detected shot.
        
        Returns:
            List[dict]: List of dictionaries containing detailed timing for each shot.
        """
        if not self.video_info or not self.shot_boundaries:
            return []
        
        fps = self.video_info['fps']
        
        # Define shot intervals. 
        # First shot starts at 0. Last shot ends at the last frame.
        shot_starts = [0] + self.shot_boundaries
        shot_ends = self.shot_boundaries + [self.video_info['frame_count']]
        
        durations = []
        for idx, (start, end) in enumerate(zip(shot_starts, shot_ends)):
            durations.append({
                'shot_id': idx + 1,
                'start_frame': int(start),
                'end_frame': int(end),
                'duration_frames': int(end - start),
                'duration_seconds': float((end - start) / fps),
                'start_time': float(start / fps),
                'end_time': float(end / fps)
            })
        
        return durations


def main():
    # Command line interface definition
    parser = argparse.ArgumentParser(description='Video Shot Boundary Detection')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', default='data/results', help='Output directory')
    parser.add_argument('--method', '-m', default='transnet', 
                       choices=['transnet', 'histogram', 'frame_diff', 'adaptive'],
                       help='Detection method')
    parser.add_argument('--threshold', '-t', type=float, default=None,
                       help='Custom threshold (default: auto)')
    parser.add_argument('--resize', '-r', type=int, nargs=2, default=None,
                       help='Resize frames to (width height) for faster processing')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Video file not found: {args.input}")
        return
    
    # Initialize detector
    detector = ShotDetector(method=args.method, threshold=args.threshold)
    
    # Parse resize argument if present
    resize = tuple(args.resize) if args.resize else None
    
    # Run detection
    shot_boundaries = detector.detect(args.input, resize=resize)
    
    # Save results
    detector.save_results(args.output, args.input)
    
    # Print summary
    print("\n" + "="*60)
    print(f"Shot Detection Completed!")
    print(f"Method: {args.method}")
    print(f"Total Shots: {len(shot_boundaries) + 1}")
    print(f"Results: {args.output}")
    print("="*60)


if __name__ == '__main__':
    main()
