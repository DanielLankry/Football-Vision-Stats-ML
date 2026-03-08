<div id="top">

<div align="center">

# Football-Vision-Stats

<em>AI-powered football (soccer) analytics system that extracts player statistics from match videos using deep learning.</em>

<br>

</div>

<br>

<img width="1908" height="910" alt="image" src="https://github.com/user-attachments/assets/6fdda3c1-24d2-4931-bf62-283ade8c65d7" />

<br>

-----

## Overview

Football-Vision-Stats is a complete deep learning pipeline for football match video analysis. It detects players, goalkeepers, referees, and the ball, tracks them across frames, classifies teams by jersey color, compensates for camera movement, and computes real-world speed and distance statistics — all from a single Jupyter notebook.

**Academic final project** for a Deep Learning course (HIT, 2025-2026).

**Key Capabilities:**

- Detect and track players, goalkeepers, referees, and the ball across video frames
- Classify players into teams using two-stage K-Means jersey color clustering
- Compensate for camera movement with Lucas-Kanade optical flow
- Map pixel coordinates to real-world field coordinates via perspective transformation
- Calculate per-player speed and distance covered
- Generate fully annotated output videos with statistics overlay, possession bar, and match stats panel
- Compare base vs. fine-tuned model performance side-by-side

**Built With:**

- **Language**: Python 3.8+
- **Object Detection**: [ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv11
- **Multi-Object Tracking**: [supervision](https://github.com/roboflow/supervision) — ByteTrack
- **Computer Vision**: OpenCV
- **Data Processing**: NumPy, Pandas
- **Clustering**: scikit-learn (K-Means)
- **Visualization**: Matplotlib
- **Environment**: Jupyter Notebook / Google Colab

-----

## Pipeline Architecture

```
Input Video
    │
    ▼
┌─────────────────────┐
│  YOLOv11 Detection  │  ← Fine-tuned on custom football dataset (4 classes)
│  conf=0.1, batch=20 │
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  ByteTrack Tracking  │  ← Persistent IDs across frames
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Team Classification │  ← Two-stage K-Means on jersey crops (top-half)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Ball Interpolation  │  ← Pandas interpolation for missing detections
│  & Possession        │  ← Closest player within 50px threshold
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Camera Compensation │  ← Lucas-Kanade optical flow on field edges
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Perspective Transform│  ← 4-point homography → 68m × 23.32m field
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Speed & Distance    │  ← 5-frame sliding window at 24 fps
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Video Annotation    │  ← Ellipses, IDs, stats, possession bar
└─────────────────────┘
          ▼
    Output Video (AVI)
```

-----

## Notebook Structure

The notebook (`Football-Vision-Stats.ipynb`) is organized into 4 parts with 65 cells:

| Part | Section | Description |
|------|---------|-------------|
| **Part 1** | Model Training | Optional Google Colab section — downloads Roboflow dataset and fine-tunes YOLOv11 |
| **Part 2** | Setup & Installation | Package installation, imports, configuration, file paths |
| **Part 3** | Model Comparison | Side-by-side comparison of base COCO model vs. fine-tuned model on the same frame |
| **Part 4** | Full Pipeline | End-to-end processing: detection → tracking → teams → ball → camera → speed → output video |

Helper functions are organized into clear sections: Video I/O, Detection & Tracking, Drawing & Annotation, Team Classification, Ball Tracking, Camera Movement, and Speed/Distance.

-----

## Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/DanielLankry/Football-Vision-Stats-ML.git
   cd Football-Vision-Stats-ML
   ```

2. **Install dependencies**:

   ```bash
   pip install ultralytics supervision opencv-python numpy pandas scikit-learn matplotlib roboflow
   ```

3. **Prepare model weights**:
   - Place your fine-tuned `best.pt` in `Trained_Models/` — or train your own using Part 1 of the notebook on Google Colab.

### Usage

1. Open `Football-Vision-Stats.ipynb` in Jupyter or Google Colab
2. Set your video path and model paths in the **Configuration** cell
3. Run cells sequentially — cached `.pkl` files skip expensive recomputation
4. Output video is saved to `runs/output_videos/`

-----

## Algorithm Details

### Team Color Classification (Two-Stage K-Means)

**Stage 1 — Per Player:** Crop the top half of each player's bounding box (jersey area). Run K-Means (k=2) to separate jersey pixels from background. The non-background cluster center is the player's jersey color.

**Stage 2 — Global:** Collect all player jersey colors and run K-Means (k=2) again to cluster them into two teams.

$$J = \sum_{i=1}^{n} \sum_{j=1}^{k} \|x_i^{(j)} - c_j\|^2$$

### Camera Movement Compensation

Uses `cv2.calcOpticalFlowPyrLK` to track good features (corners, edges) between consecutive frames, estimating camera pan and tilt. Player positions are adjusted by subtracting the estimated camera movement.

### Perspective Transformation

A 4-point homography matrix maps pixel coordinates to real-world field coordinates (68m × 23.32m). Combined with camera movement compensation, this yields accurate real-world positions for speed and distance calculations.

-----

## Dataset

Source: [Roboflow](https://universe.roboflow.com/) — `football-players-detection` dataset.
- **663 images** with YOLO-format annotations
- **4 classes**: ball, goalkeeper, player, referee
- **License**: CC BY 4.0

-----

## Caching

Detection and tracking results are cached as `.pkl` files in `runs/output_videos/`. Delete these files to force recomputation when changing models or parameters.

-----

## Project Structure

```
.
├── Football-Vision-Stats.ipynb          # Main notebook — complete pipeline (65 cells)
├── Trained_Models/
│   ├── best.pt                          # Fine-tuned YOLOv11 checkpoint
│   └── last.pt                          # Last epoch checkpoint
├── runs/
│   └── output_videos/                   # Output videos and cached .pkl data
├── Football Advanced stats with python/ # Reference materials
└── README.md
```

-----

## License

This project is licensed under the MIT License.

-----

## Contact

- **Author**: Daniel Lankry
- **Project Link**: [https://github.com/DanielLankry/Football-Vision-Stats-ML](https://github.com/DanielLankry/Football-Vision-Stats-ML)

-----

## Acknowledgments

- [ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv11 object detection
- [supervision](https://github.com/roboflow/supervision) — ByteTrack tracking and annotation
- [OpenCV](https://opencv.org/) — Video and image processing
- [Roboflow](https://roboflow.com/) — Dataset hosting and management

<div align="center">

<br>
<a href="#top">Back to Top</a>
<br>

</div>
