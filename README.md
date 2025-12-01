# Video Scene Boundary Detection

**Student:** Serkan Efe Camoğlu  
**Course:** CSE495 - Graduation Project

## 📌 Project Description
This project implements a system for detecting scene boundaries in videos. It is designed as a two-phase pipeline:
1.  **Shot Boundary Detection** (Completed) - Detects individual camera shots.
2.  **Scene Detection** (Implemented) - Groups shots into semantic scenes using deep learning features and optimal sequential grouping.

## ✅ Completed Features

### Phase 1: Shot Detection
-   ✅ **TransNet V2** integration (Deep Learning based, >95% accuracy).
-   ✅ Alternative methods: Histogram-based, Frame Difference, Adaptive Threshold.
-   ✅ **FFmpeg integration** for efficient video processing.
-   ✅ **Visualization**: Shot boundary plots, keyframe extraction, shot summary images.

### Phase 2: Scene Detection
-   ✅ **Semantic Feature Extraction**: Uses **OpenAI CLIP (ViT-B/32)** to extract semantic features from shot keyframes.
-   ✅ **Optimal Sequential Grouping (OSG)**: Uses the **H_nrm** algorithm (dynamic programming) to group shots into scenes based on visual coherence.
-   ✅ **Automatic Scene Count Estimation**: Uses Singular Value Decomposition (SVD) on the similarity matrix to estimate the optimal number of scenes (K).
-   ✅ **Output**: JSON reports detailing scene start/end times and representative keyframes.

## 📂 Project Structure & File Descriptions

This section provides a detailed overview of the files in the repository.

### Source Code (`src/`)
*   **`src/shot_detection.py`**: Main entry point for shot detection. Orchestrates the process and handles saving results.
*   **`src/scene_detection.py`**: Main entry point for scene detection. Integrates CLIP features and the OSG algorithm to group shots.
*   **`src/feature_extraction.py`**: Handles loading the OpenAI CLIP model and extracting semantic embeddings from images.
*   **`src/transnet_detector.py`**: Wrapper for the TransNet V2 model.
*   **`src/utils/video_utils.py`**: Low-level video processing functions (frame reading, histogram calculation).
*   **`src/utils/visualization.py`**: Functions for generating visual outputs (plots, grids).

### Scripts & Tools
*   **`test_scene_detection.py`**: Automated test script for the full pipeline (Shot Detection -> Scene Detection).
*   **`test_shot_detection.py`**: Test script for the shot detection phase.
*   **`example_usage.py`**: Code examples for using the `ShotDetector` class.
*   **`install_transnet.py`**: Helper to install TransNet V2 from GitHub.

### Data & Configuration
*   **`data/videos/`**: Input video files.
*   **`data/results/`**: Output results (plots, JSON reports, keyframes).
*   **`requirements.txt`**: Python dependencies.
*   **`cluster_dp_algo/`**: Submodule containing the Optimal Sequential Grouping (OSG) algorithm implementation.

## 🚀 Installation

### Prerequisites
-   Python 3.8+
-   Git
-   FFmpeg (Required)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Install TransNet V2
TransNet V2 requires a manual install from GitHub (model size ~100MB).
```bash
python install_transnet.py
```

### Step 3: Install FFmpeg
Windows (via Chocolatey):
```powershell
choco install ffmpeg
```

## 🛠 Usage

### 1. Full Pipeline Test (Recommended)
Runs both Shot Detection and Scene Detection on a sample video.
```bash
python test_scene_detection.py
```

### 2. Shot Detection Only
```bash
python src/shot_detection.py --input data/videos/sample_video.mp4 --method transnet
```

### 3. Scene Detection (Programmatic)
```python
from src.scene_detection import SceneDetector

# Assuming shot_boundaries are already detected
scene_detector = SceneDetector()
scenes = scene_detector.detect_scenes(video_path, shot_boundaries, output_dir)
```

## 📊 Method Details

| Component | Method | Description | Status |
|-----------|--------|-------------|--------|
| **Shot Detection** | **TransNet V2** | Deep Learning (CNN). Detects hard cuts & gradual transitions. | ✅ Ready |
| **Feature Extraction** | **OpenAI CLIP** | Vision Transformer (ViT). Extracts semantic meaning from images. | ✅ Ready |
| **Scene Clustering** | **OSG (H_nrm)** | Optimal Sequential Grouping using Dynamic Programming. Preserves temporal order. | ✅ Ready |

## 📚 References
-   **TransNet V2**: [Repo](https://github.com/soCzech/TransNetV2)
-   **CLIP**: [OpenAI](https://github.com/openai/CLIP)
-   **OSG / Cluster DP Algo**: [Repo](https://github.com/pyscenedetect/cluster-dp-algo)

---
**Last Updated:** December 1, 2025
