"""
TransNet V2 Shot Boundary Detection Wrapper

This module serves as an interface to the TransNet V2 library.
TransNet V2 is a state-of-the-art deep learning model specifically designed
for shot boundary detection. It uses Dilated 3D CNNs to capture temporal
dependencies and transitions.

Reference: https://github.com/soCzech/TransNetV2
"""

import os
import numpy as np
from typing import List, Tuple, Optional
import cv2
from tqdm import tqdm

# Attempt to import TransNetV2. The import is wrapped in a try-except block
# to allow the rest of the project to function even if this specific dependency is missing.
try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO/WARNING logs
    from transnetv2 import TransNetV2
    TRANSNET_AVAILABLE = True
except ImportError:
    TRANSNET_AVAILABLE = False
    print("WARNING: TransNet V2 not installed. Install via 'python install_transnet.py'")


class TransNetDetector:
    """
    Wrapper class for the TransNet V2 model.
    
    Handles:
    - Model initialization and weight loading.
    - Interfacing with the `transnetv2` library.
    - Parsing the model's output into a standardized format.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize the TransNet detector.

        Args:
            threshold: The confidence score (0.0 to 1.0) required to classify a frame as a shot boundary.
                       The default of 0.5 is recommended by the TransNet V2 authors.
        """
        if not TRANSNET_AVAILABLE:
            raise ImportError(
                "TransNet V2 is not installed! To install:\n"
                "python install_transnet.py"
            )
        
        self.threshold = threshold
        self.model = None
        self.predictions = None
        self.shot_boundaries = []
        
    def _load_model(self):
        """
        Lazy-loads the TransNet V2 model.
        
        The model weights (~100MB) are downloaded automatically by the library
        on the first run and cached locally.
        """
        if self.model is None:
            print("Loading TransNet V2 model...")
            print("(Model will be downloaded on first use, ~100MB)")
            self.model = TransNetV2()
            print("✓ Model loaded!")
    
    def detect(self, video_path: str) -> Tuple[List[int], np.ndarray]:
        """
        Performs shot boundary detection on a video file.
        
        Args:
            video_path: Path to the input video file.
            
        Returns:
            Tuple containing:
            - shot_boundaries: List of frame indices where shots change.
            - single_frame_predictions: Raw confidence scores (0-1) for each frame.
        """
        self._load_model()
        
        print(f"\nStarting shot detection with TransNet V2...")
        print(f"Video: {video_path}")
        print("Running model prediction...")
        print("(TransNet V2 is processing the video internally...)")
        
        # TransNet V2's predict_video method handles reading the video file 
        # efficiently using ffmpeg-python internally.
        result = self.model.predict_video(video_path)
        
        # Parse the output structure, which can vary slightly based on usage.
        # Typically returns: (single_frame_predictions, all_frame_predictions)
        # Or sometimes: (frames, single_frame_predictions, all_frame_predictions)
        
        if isinstance(result, tuple):
            
            if len(result) >= 2:
                # Heuristic to check if the first element is raw video frames (4D array)
                if hasattr(result[0], 'ndim') and result[0].ndim == 4:
                    # If [0] is frames, then [1] is the prediction we want
                    single_frame_predictions = result[1]
                else:
                    # Otherwise, [0] is the prediction
                    single_frame_predictions = result[0]
            else:
                single_frame_predictions = result[0]
        else:
            single_frame_predictions = result
        
        # TransNet outputs shape (N, 1) for predictions, we simplify to (N,)
        if hasattr(single_frame_predictions, 'ndim') and single_frame_predictions.ndim == 2:
            single_frame_predictions = single_frame_predictions[:, 0]  # Take first column
        
        # Convert continuous probability scores into discrete boundary indices
        shot_boundaries = self._find_boundaries(single_frame_predictions)
        
        self.predictions = single_frame_predictions
        self.shot_boundaries = shot_boundaries
        
        print(f"✓ {len(shot_boundaries)} shot boundaries detected!")
        
        return shot_boundaries, single_frame_predictions
    
    def _find_boundaries(self, predictions: np.ndarray) -> List[int]:
        """
        Converts raw prediction scores into shot boundary frame indices.
        
        Args:
            predictions: 1D array of float scores (0.0 to 1.0) for each frame.
            
        Returns:
            List of frame indices where score >= threshold.
        """
        boundaries = []
        
        # Ensure input is a 1D array
        if hasattr(predictions, 'ndim') and predictions.ndim > 1:
            predictions = predictions.flatten()
        
        for i, score in enumerate(predictions):
            # Defensive type casting to ensure we have a standard float
            if isinstance(score, np.ndarray):
                score = float(score[0])
            else:
                score = float(score)
            
            if score >= self.threshold:
                # Debouncing: If a boundary was detected very recently (within 10 frames),
                # we assume this is part of the same transition and skip it to avoid duplicates.
                if not boundaries or (i - boundaries[-1] > 10):
                    boundaries.append(i)
        
        return boundaries
