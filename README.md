# Video Scene Boundary Detection

**Student:** Serkan Efe Camoğlu  
**Course:** CSE495 - Graduation Project

## 📌 Project Description
This project implements a system for detecting scene boundaries in videos. It is designed as a two-phase pipeline:
1.  **Shot Boundary Detection** (Completed)
2.  **Scene Detection** (Planned)

## ✅ Completed Features (Phase 1)
-   ✅ **Shot Boundary Detection** using **TransNet V2** (Deep Learning based, >95% accuracy).
-   ✅ Alternative methods implemented: Histogram-based, Frame Difference, Adaptive Threshold.
-   ✅ **FFmpeg integration** for efficient video processing.
-   ✅ **Visualization & Reporting**: Shot boundary plots, keyframe extraction, shot summary images, and JSON reports.
-   ✅ **Test Coverage**: Verified with test videos achieving 100% accuracy on sample data.

## 📂 Project Structure & File Descriptions

This section provides a detailed overview of the files in the repository and their specific roles.

### Source Code (`src/`)
*   **`src/shot_detection.py`**: The main entry point for the shot detection logic. It defines the `ShotDetector` class which orchestrates the process, switches between different algorithms (TransNet, Histogram, etc.), and handles saving results.
*   **`src/transnet_detector.py`**: A specialized wrapper class for the TransNet V2 model. It handles model loading, interacts with the deep learning backend, and parses the raw predictions into usable shot boundaries.
*   **`src/utils/video_utils.py`**: Contains low-level video processing functions. This includes reading video frames, calculating color histograms, and computing pixel-level differences between frames.
*   **`src/utils/visualization.py`**: Responsible for generating visual outputs. It includes functions to plot detection graphs, create shot summary grids, and save individual keyframes.

### Scripts & Tools
*   **`test_shot_detection.py`**: An automated test script that verifies the installation of all dependencies and runs a full detection pipeline on a sample video to ensure everything is working correctly.
*   **`example_usage.py`**: Provides clear, copy-pasteable code examples demonstrating how to use the library in different ways (e.g., basic usage, comparing methods, custom thresholds).
*   **`install_transnet.py`**: A helper script to automate the installation of TransNet V2, which is not available on PyPI and must be installed from GitHub. It also handles installing PyTorch dependencies.

### Data & Configuration
*   **`data/videos/`**: Directory for storing input video files.
*   **`data/results/`**: Directory where the output results (plots, JSON reports, keyframes) are saved.
*   **`requirements.txt`**: Lists all Python libraries required to run the project.
*   **`docs/`**: Contains reference materials, academic papers, and presentation slides related to the project.

## 🚀 Installation

### Prerequisites
-   Python 3.8+
-   Git
-   FFmpeg (Required for video processing)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Install TransNet V2
TransNet V2 requires a manual install from GitHub (model size ~100MB).
```bash
python install_transnet.py
```
*Alternatively:* `pip install git+https://github.com/soCzech/TransNetV2.git`

### Step 3: Install FFmpeg
If you are on Windows, you can use Chocolatey:
```powershell
choco install ffmpeg
```
*Ensure `ffmpeg` is added to your system PATH.*

## 🛠 Usage

### Basic Usage (TransNet V2 - Recommended)
```bash
python src/shot_detection.py --input data/videos/sample_video.mp4 --method transnet
```

### Alternative Methods
```bash
# Histogram-based (Fast, lower accuracy)
python src/shot_detection.py --input data/videos/sample_video.mp4 --method histogram

# Frame Difference
python src/shot_detection.py --input data/videos/sample_video.mp4 --method frame_diff
```

### Run Automated Test
```bash
python test_shot_detection.py
```

## 📊 Shot Detection Methods

| Method | Description | Accuracy | Speed |
|--------|-------------|----------|-------|
| **TransNet V2** ⭐ | Deep Learning (CNN). Detects hard cuts & gradual transitions. | High (>95%) | Moderate |
| **Histogram** | Compares color histograms between frames. | Medium (~80%) | Fast |
| **Frame Diff** | Pixel-level difference analysis. | Low-Medium | Fast |
| **Adaptive** | Sliding window thresholding. | Medium | Moderate |

## 🔮 Future Work (Phase 2: Scene Detection)
The next phase will group shots into semantic scenes.

1.  **Feature Extraction**: Extract visual features (ResNet/VGG) from shot keyframes.
2.  **Clustering**: Group similar shots using K-Means or similar algorithms.
3.  **Boundary Detection**: Identify transitions between clusters as scene boundaries.

## 📚 References
-   **TransNet V2**: [Repo](https://github.com/soCzech/TransNetV2)
-   **PySceneDetect**: [Repo](https://github.com/Breakthrough/PySceneDetect)
-   *Shot Boundary Detection - Fundamental Concepts* (see `docs/`)

---
**Last Updated:** November 8, 2025
