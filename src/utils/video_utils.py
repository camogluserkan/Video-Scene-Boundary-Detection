"""
Video Processing Utilities Module

This module contains low-level helper functions for reading, manipulating, 
and analyzing video frames. It abstracts away the details of the OpenCV 
library to provide a clean interface for the main detection logic.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List


def get_video_info(video_path: str) -> dict:
    """
    Retrieves metadata for a given video file.
    
    This function opens the video file and extracts properties such as 
    frame rate (FPS), total frame count, resolution, and duration.
    
    Args:
        video_path: Full path to the video file.
        
    Returns:
        dict: A dictionary with keys: 'fps', 'frame_count', 'width', 'height', 'duration'.
        
    Raises:
        ValueError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info


def read_video_frames(video_path: str, 
                      resize: Optional[Tuple[int, int]] = None,
                      max_frames: Optional[int] = None) -> np.ndarray:
    """
    Reads all frames (or a subset) from a video into a NumPy array.
    
    Warning: Loading a long video entirely into memory can consume a lot of RAM.
    Use `max_frames` or process sequentially for large files.
    
    Args:
        video_path: Path to the video file.
        resize: Optional tuple (width, height) to resize each frame.
        max_frames: Maximum number of frames to read.
        
    Returns:
        np.ndarray: A 4D array of shape (num_frames, height, width, 3).
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if resize:
            frame = cv2.resize(frame, resize)
            
        frames.append(frame)
        frame_count += 1
        
        if max_frames and frame_count >= max_frames:
            break
    
    cap.release()
    return np.array(frames)


def extract_frames_at_indices(video_path: str, 
                              frame_indices: List[int],
                              resize: Optional[Tuple[int, int]] = None) -> List[Optional[np.ndarray]]:
    """
    Efficiently extracts specific frames from a video based on their indices.
    
    Instead of reading the entire video, this seeks directly to the desired frame numbers.
    
    Args:
        video_path: Path to the video file.
        frame_indices: List of integer indices of the frames to extract.
        resize: Optional tuple (width, height) to resize the extracted frames.
        
    Returns:
        List[Optional[np.ndarray]]: A list of frames. Elements can be None if reading failed.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frames = []
    # Sorting indices allows for forward sequential seeking, which is faster/more reliable
    frame_indices = sorted(frame_indices)
    
    for target_idx in frame_indices:
        # Seek to the specific frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        
        if ret:
            if resize:
                frame = cv2.resize(frame, resize)
            frames.append(frame)
        else:
            frames.append(None)
    
    cap.release()
    return frames


def save_frame(frame: np.ndarray, output_path: str) -> bool:
    """
    Saves a single frame to disk as an image file.
    
    Args:
        frame: The image data (NumPy array).
        output_path: Destination path (e.g., "output/frame_001.jpg").
        
    Returns:
        bool: True if successful, False otherwise.
    """
    return cv2.imwrite(output_path, frame)


def calculate_histogram(frame: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Calculates the color histogram for a given frame.
    
    The function computes histograms for Blue, Green, and Red channels separately,
    normalizes them, and concatenates them into a single feature vector.
    
    Args:
        frame: Input image (BGR format).
        bins: Number of bins per channel (default 256).
        
    Returns:
        np.ndarray: Flattened 1D array of size (3 * bins).
    """
    histograms = []
    
    for i in range(3):  # Iterate over B, G, R channels
        # Calculate histogram for the i-th channel
        hist = cv2.calcHist([frame], [i], None, [bins], [0, 256])
        # Normalize to make it scale-invariant (independent of image resolution)
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)
    
    return np.concatenate(histograms)


def frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calculates the normalized average pixel difference between two frames.
    
    This is a simple metric for motion or shot change detection. Frames are
    converted to grayscale first to reduce computational complexity.
    
    Args:
        frame1: First frame image.
        frame2: Second frame image.
        
    Returns:
        float: A value between 0.0 (identical) and 1.0 (completely different).
    """
    # Convert to grayscale to focus on luminance changes
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute pixel-wise difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Compute the mean difference and normalize to [0, 1] range
    # 255.0 is the maximum possible pixel value
    mean_diff = np.mean(diff) / 255.0
    
    return mean_diff


def histogram_difference(hist1: np.ndarray, hist2: np.ndarray, 
                         method: str = 'chi-square') -> float:
    """
    Calculates the statistical difference between two histograms.
    
    Args:
        hist1: First histogram feature vector.
        hist2: Second histogram feature vector.
        method: The comparison metric.
                - 'chi-square': Common for histograms, lower is better match (here used as distance).
                - 'correlation': 1 is perfect match, -1 is inverse. We invert it to 0-1 distance.
                - 'intersection': Measures overlap. We invert it to represent distance.
        
    Returns:
        float: The calculated distance/difference score.
    """
    if method == 'chi-square':
        # Chi-Square is a standard metric for histogram comparison
        return cv2.compareHist(hist1.astype(np.float32), 
                              hist2.astype(np.float32), 
                              cv2.HISTCMP_CHISQR)
    elif method == 'correlation':
        # Invert correlation so that higher values mean "more different"
        return 1 - cv2.compareHist(hist1.astype(np.float32), 
                                   hist2.astype(np.float32), 
                                   cv2.HISTCMP_CORREL)
    elif method == 'intersection':
        # Invert intersection so that higher values mean "less overlap" (more different)
        return 1 - cv2.compareHist(hist1.astype(np.float32), 
                                   hist2.astype(np.float32), 
                                   cv2.HISTCMP_INTERSECT)
    else:
        raise ValueError(f"Unknown method: {method}")
