"""
Shot Detection Example Usage Script

This script demonstrates how to use the `ShotDetector` class in various scenarios.
It serves as a practical guide for developers to understand the API and capabilities
of the library.

Included Examples:
1. Basic Usage: Running TransNet V2 on a video and saving results.
2. Method Comparison: Running multiple algorithms side-by-side to compare their output.
3. Custom Configuration: Tweaking parameters like thresholds for specific needs.
"""

import os
from src.shot_detection import ShotDetector

def example_1_basic_usage():
    """
    Example 1: Basic Usage with TransNet V2
    
    This is the recommended way to use the library for most users.
    It uses the state-of-the-art TransNet V2 model which offers high accuracy
    without manual tuning.
    """
    print("=" * 60)
    print("EXAMPLE 1: Shot Detection with TransNet V2 (Highest Accuracy)")
    print("=" * 60)
    
    # Define the path to the video file
    video_path = "data/videos/sample_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found!")
        print("Please add a video file to 'data/videos/'.")
        return
    
    # Initialize the ShotDetector with 'transnet' method.
    # Threshold 0.5 is the standard cutoff for this model.
    detector = ShotDetector(method='transnet', threshold=0.5)
    
    # Run detection.
    # Note: TransNet handles frame resizing internally for performance.
    shot_boundaries = detector.detect(video_path)
    
    # Save all results (plots, images, reports) to a specific folder.
    output_folder = 'data/results/example1'
    detector.save_results(output_folder, video_path)
    
    print(f"\n✓ Example 1 completed! Detected shots: {len(shot_boundaries) + 1}")
    print(f"Results saved to: {output_folder}")


def example_2_compare_methods():
    """
    Example 2: Method Comparison
    
    This example runs three different detection algorithms on the same video
    and generates a stacked plot to compare their behavior. This is useful
    for understanding the strengths and weaknesses of each approach.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Comparing Different Methods")
    print("=" * 60)
    
    video_path = "data/videos/sample_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found!")
        return
    
    # List of methods to test
    methods = ['transnet', 'histogram', 'adaptive']
    results = {}
    
    for method in methods:
        print(f"\nTesting {method.upper()} method...")
        
        detector = ShotDetector(method=method)
        
        # Optimization: Use resizing for non-DL methods to speed up processing significantly.
        # TransNet uses its own optimized pipeline, so we don't pass resize to it.
        if method == 'transnet':
            shot_boundaries = detector.detect(video_path)
        else:
            shot_boundaries = detector.detect(video_path, resize=(320, 180))
        
        # Store results for comparison
        results[method] = {
            'differences': detector.differences,
            'boundaries': shot_boundaries,
            'threshold': detector.threshold
        }
        
        print(f"  → {len(shot_boundaries) + 1} shots detected")
    
    # Generate and save the comparison plot
    from src.utils.visualization import compare_methods
    compare_methods(results, save_path='data/results/method_comparison.png')
    
    print("\n✓ Example 2 completed! Comparison plot: data/results/method_comparison.png")


def example_3_custom_threshold():
    """
    Example 3: Custom Thresholding
    
    This example demonstrates how to override the automatic threshold calculation.
    Increasing the threshold makes the detector less sensitive (detects fewer, more obvious cuts).
    Decreasing it makes it more sensitive (detects subtler changes, but potentially more false positives).
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Detection with Custom Threshold")
    print("=" * 60)
    
    video_path = "data/videos/sample_video.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found!")
        return
    
    # First run: Use automatic thresholding
    print("--- Run 1: Automatic Threshold ---")
    detector_auto = ShotDetector(method='histogram')
    detector_auto.detect(video_path, resize=(320, 180))
    auto_threshold = detector_auto.threshold
    
    print(f"Automatic threshold: {auto_threshold:.4f}")
    print(f"Detected shots: {len(detector_auto.shot_boundaries) + 1}")
    
    # Second run: Manually set a higher threshold (e.g., 50% higher)
    # This is useful if the automatic run was detecting too much noise as cuts.
    custom_threshold = auto_threshold * 1.5
    print(f"\n--- Run 2: Custom Threshold (x1.5) ---")
    print(f"Custom threshold: {custom_threshold:.4f}")
    
    detector_custom = ShotDetector(method='histogram', threshold=custom_threshold)
    detector_custom.detect(video_path, resize=(320, 180))
    
    print(f"Detected shots: {len(detector_custom.shot_boundaries) + 1}")
    
    print("\n✓ Example 3 completed!")


def download_sample_video_info():
    """
    Displays information on where to find sample videos for testing.
    """
    print("\n" + "=" * 60)
    print("SAMPLE VIDEO DOWNLOAD")
    print("=" * 60)
    print("\nYou can download free test videos from:")
    print("1. Pexels: https://www.pexels.com/videos/")
    print("2. Pixabay: https://pixabay.com/videos/")
    print("3. Videvo: https://www.videvo.net/")
    print("\nSave your video as 'data/videos/sample_video.mp4'.")
    print("\nOr via command line (if yt-dlp is installed):")
    print("  yt-dlp -f 'bestvideo[height<=480]' -o 'data/videos/sample_video.mp4' [URL]")


if __name__ == '__main__':
    # Ensure necessary directories exist
    os.makedirs('data/videos', exist_ok=True)
    os.makedirs('data/results', exist_ok=True)
    
    # Check if the sample video exists
    sample_video = "data/videos/sample_video.mp4"
    
    if not os.path.exists(sample_video):
        print("=" * 60)
        print("WARNING: Sample video not found!")
        print("=" * 60)
        download_sample_video_info()
        print("\nPlease run this script again after adding a video.")
    else:
        # Run the examples
        # You can comment/uncomment these lines to run specific examples
        
        example_1_basic_usage()
        
        # Uncomment to run comparison:
        # example_2_compare_methods()
        
        # Uncomment to run custom threshold test:
        # example_3_custom_threshold()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED!")
        print("=" * 60)
        print("\nResults are in 'data/results/' directory.")
        print("Edit this script to uncomment other examples.")
