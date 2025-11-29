"""
Visualization Utilities Module

This module is responsible for generating visual outputs from the shot detection process.
It includes functions to create time-series plots of frame differences, summary grids
of shot keyframes, and comparison charts for multiple methods.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import os


def plot_shot_boundaries(differences: np.ndarray, 
                        shot_boundaries: List[int],
                        threshold: float,
                        title: str = "Shot Boundary Detection",
                        save_path: Optional[str] = None):
    """
    Generates a time-series plot of frame differences and marks detected boundaries.
    
    The plot shows:
    - Blue line: The calculated difference metric (or confidence score) for each frame.
    - Red dashed line: The threshold value used for detection.
    - Green vertical lines: The frames identified as shot boundaries.
    
    Args:
        differences: 1D array of difference scores for each frame.
        shot_boundaries: List of frame indices where shots start/end.
        threshold: The numeric threshold used to decide cut points.
        title: Title of the plot graph.
        save_path: If provided, the plot is saved to this file path instead of shown.
    """
    plt.figure(figsize=(15, 6))
    
    # Plot the continuous signal (difference metric or prediction score)
    plt.plot(differences, label='Frame Differences', linewidth=1)
    
    # Draw the threshold line for visual reference
    plt.axhline(y=threshold, color='r', linestyle='--', 
                label=f'Threshold ({threshold:.4f})', linewidth=2)
    
    # Mark each detected boundary with a vertical line
    for boundary in shot_boundaries:
        plt.axvline(x=boundary, color='g', linestyle='-', 
                   alpha=0.3, linewidth=1)
    
    plt.xlabel('Frame Number')
    plt.ylabel('Difference Value')
    plt.title(f'{title}\nDetected Shots: {len(shot_boundaries)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    plt.close()  # Close the figure to prevent memory leaks when running in loops


def create_shot_summary(video_path: str,
                       shot_boundaries: List[int],
                       output_path: str,
                       grid_cols: int = 4):
    """
    Creates a single image composed of keyframes from every detected shot.
    
    This provides a quick visual overview of the entire video content.
    The middle frame of each shot is selected as the representative keyframe.
    
    Args:
        video_path: Path to the source video.
        shot_boundaries: List of frame indices marking shot changes.
        output_path: File path where the summary image will be saved.
        grid_cols: Number of columns in the output grid layout.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Calculate start and end frames for each shot to find the middle
    keyframes = []
    shot_starts = [0] + shot_boundaries
    shot_ends = shot_boundaries + [int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
    
    for start, end in zip(shot_starts, shot_ends):
        middle_frame = (start + end) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        
        if ret:
            # Resize frame to a standard small size for the grid
            frame = cv2.resize(frame, (320, 180))
            # Overlay shot information on the frame
            cv2.putText(frame, f"Shot {len(keyframes)+1}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {middle_frame}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            keyframes.append(frame)
    
    cap.release()
    
    if not keyframes:
        print("No keyframes extracted.")
        return

    # Calculate grid dimensions
    n_shots = len(keyframes)
    grid_rows = (n_shots + grid_cols - 1) // grid_cols
    
    # Create a blank black canvas
    h, w = 180, 320
    grid = np.zeros((grid_rows * h, grid_cols * w, 3), dtype=np.uint8)
    
    # Paste each keyframe into the grid
    for idx, frame in enumerate(keyframes):
        row = idx // grid_cols
        col = idx % grid_cols
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = frame
    
    # Save the final composition
    cv2.imwrite(output_path, grid)
    print(f"Shot summary saved: {output_path}")


def save_shot_keyframes(video_path: str,
                       shot_boundaries: List[int],
                       output_dir: str):
    """
    Extracts and saves the middle frame of each shot as a separate image file.
    
    This is useful for the subsequent Scene Detection phase, where clustering
    algorithms will operate on these keyframes.
    
    Args:
        video_path: Path to the source video.
        shot_boundaries: List of detected shot boundaries.
        output_dir: Directory where keyframes will be stored.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    shot_starts = [0] + shot_boundaries
    shot_ends = shot_boundaries + [int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
    
    saved_count = 0
    for shot_idx, (start, end) in enumerate(zip(shot_starts, shot_ends)):
        middle_frame = (start + end) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        
        if ret:
            output_path = os.path.join(output_dir, f"shot_{shot_idx+1:03d}_frame_{middle_frame}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
    
    cap.release()
    print(f"{saved_count} shot keyframes saved to: {output_dir}")


def compare_methods(results_dict: dict, 
                   save_path: Optional[str] = None):
    """
    Creates a stacked plot comparing results from multiple detection methods.
    
    This is helpful for analyzing how different algorithms (e.g., TransNet vs Histogram)
    perform on the same video.
    
    Args:
        results_dict: A dictionary structured as:
                      {
                          'method_name': {
                              'differences': np.ndarray,
                              'boundaries': List[int],
                              'threshold': float
                          },
                          ...
                      }
        save_path: Optional path to save the comparison chart.
    """
    n_methods = len(results_dict)
    fig, axes = plt.subplots(n_methods, 1, figsize=(15, 4*n_methods))
    
    # Ensure axes is iterable even if there's only one method
    if n_methods == 1:
        axes = [axes]
    
    for idx, (method_name, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        
        differences = results['differences']
        boundaries = results['boundaries']
        threshold = results.get('threshold', 0)
        
        # Plot the signal
        ax.plot(differences, label='Frame Differences', linewidth=1)
        if threshold:
            ax.axhline(y=threshold, color='r', linestyle='--', 
                      label=f'Threshold ({threshold:.4f})', linewidth=2)
        
        # Mark boundaries
        for boundary in boundaries:
            ax.axvline(x=boundary, color='g', linestyle='-', 
                      alpha=0.3, linewidth=1)
        
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Difference Value')
        ax.set_title(f'{method_name} - Detected Shots: {len(boundaries)}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved: {save_path}")
    
    plt.close()
