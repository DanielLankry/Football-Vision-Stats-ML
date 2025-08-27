<div id="top">

<div align="center">

# Football-Vision-Stats

<em>A computer vision project for football analytics, using object detection, tracking, and team color clustering to analyze video footage.</em>

<br>

</div>

<br>

<div align="center">
<img src="https://i.imgur.com/gK1N42i.png" alt="Sample annotated video frame" width="600">
</div>

<br>

-----

## 📄 Table of Contents

  - [Overview](#overview)
  - [Features](#features)
  - [Getting Started](#getting-started)
      - [Prerequisites](#prerequisites)
      - [Installation](#installation)
  - [Usage](#usage)
  - [Algorithm Explanation](#algorithm-explanation)
  - [Project Structure](#project-structure)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)
  - [Acknowledgments](#acknowledgments)
  - [Back to Top](#back-to-top)

-----

## ✨ Overview

Football-Vision-Stats is a computer vision project designed to analyze football video footage. It uses a combination of deep learning models and traditional computer vision techniques to detect players, the ball, and referees, track their movements, assign them to teams, and calculate key statistics. This project is ideal for sports analysts, data scientists, and students interested in applying machine learning to real-world problems.

🎯 **Project Goals:**

  - Accurately detect and track players, the ball, and referees in a video.
  - Differentiate between teams using a color clustering algorithm.
  - Calculate and display player-specific data like speed and distance covered.
  - Provide a clear, reproducible workflow for video analysis.

🛠️ **Built With:**

  - **Language**: Python 3.8+
  - **Machine Learning**: `ultralytics` (YOLOv8)
  - **Computer Vision**: `supervision`, `opencv`
  - **Data Handling**: `pandas`, `numpy`
  - **Clustering**: `scikit-learn`
  - **Development**: `jupyter` notebook, `git`

-----

## ✨ Features

This project provides a robust framework for video analysis with the following key features:

  - 🚀 **Object Detection**: Uses a pre-trained YOLOv8 model to identify players, goalkeepers, referees, and the ball.
  - 🔄 **Multi-Object Tracking**: Implements the `ByteTrack` algorithm from the `supervision` library to track objects across video frames.
  - 🎨 **Team Color Clustering**: Utilizes K-means clustering on the top half of player bounding boxes to identify and assign team colors.
  - ⚽ **Ball Detection & Interpolation**: Detects the ball and uses interpolation to fill in missing positions, ensuring continuous tracking.
  - 📐 **Distance and Speed Calculation**: Calculates the distance covered and speed of each player by transforming pixel coordinates to real-world measurements.
  - 🖼️ **Video Annotation**: Annotates video frames with bounding boxes, team colors, player IDs, and player statistics.
  - 💾 **Persistence**: Saves tracked object data and camera movement information to disk (`.pkl` files) to avoid redundant processing.

-----

## 🚀 Getting Started

Follow these steps to set up the project locally.

### ✅ Prerequisites

Ensure you have Python 3.8 or a later version installed. The project also relies on several Python packages.

  - **Python 3.8+**
  - **Git**

### ⚙️ Installation

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/lankdaniel14/Football-Vision-Stats-ML.git
    cd Football-Vision-Stats-ML
    ```

2.  **Install the required Python packages**:
    The project uses a `pyproject.toml` file or you can directly install the packages from the notebook.

    ```bash
    %pip install ultralytics
    %pip install supervision
    %pip install python-dotenv
    ```

3.  **Prepare the environment**:

      - Create a `.env` file in the root directory.
      - Add your GitHub repository link to the `.env` file to enable cloning within Google Colab (if you choose to use it for training).

    ```env
    # .env
    GITHUB_REPO_LINK_FOOTBALL_VISION_STATS="YOUR_REPO_LINK"
    ```

      - The notebook includes a command to install `python-dotenv` and load this file.

-----

## 🏎️ Usage

The project is designed to be run as a Jupyter notebook. Execute the cells in order to perform the full analysis pipeline.

1.  **Open the notebook**:
    Open the `Football-Vision-Stats.ipynb` notebook in your preferred environment (e.g., Jupyter, Google Colab).

2.  **Run the cells sequentially**:

      - The first few cells handle package installation and imports.
      - The **"Execute the cells and instructions with this Header if you want to train on Google Colab"** section contains instructions and cells for setting up and training the model in a Google Colab environment. This step is optional if you are using a pre-trained model.
      - The **"Quick Model Capabilities AFTER training"** cell demonstrates the model's ability to detect and track objects on a single frame.

    <div align="center">
    <img src="https://i.imgur.com/37V0j9o.png" alt="Visual representation of perspective transformation on the court" width="600">
    </div>

      - The **"Features Time"** section contains the core logic for the analysis.
      - Cells in the **"Reading and Saving Video"** section read the raw video and save a copy.
      - Cells under **"Tracking bounding boxes"** use the trained model to detect objects in the video frames and track them.
      - The **"Team Color separating"** section uses K-means clustering to assign team colors to players.
      - The **"Now We're going to work on ball detection"** section includes logic to interpolate missing ball positions for smoother tracking.
      - The final cells handle camera movement, perspective correction, and calculating speed and distance.

    <br>

    You may need to modify file paths in the notebook to match your local setup or specific video files.

-----

## 🧠 Algorithm Explanation

### Team Color Clustering

The project uses **K-means clustering** to determine team colors. Instead of analyzing the entire player image, it crops the **top half** of the player's bounding box. This focuses the analysis on the player's jersey, excluding the field and legs, which are often the same color for both teams. The algorithm then clusters the pixels in this top half into two groups, with the cluster centers representing the average team colors.

$$J = \sum_{i=1}^{n} \sum_{j=1}^{k} \|x_i^{(j)} - c_j\|^2$$

  * $J$: The objective function to minimize (sum of squared distances).
  * $x_i$: A pixel's color value.
  * $c_j$: The centroid of cluster $j$.
  * $k$: The number of clusters (set to 2 for the two teams).

### Camera Movement and Perspective Correction

To accurately calculate player speed and distance, the project accounts for camera movement and converts pixel coordinates to real-world coordinates.

1.  **Optical Flow**: The `cv2.calcOpticalFlowPyrLK` function is used to estimate camera motion between frames by tracking a set of **good features to track** (e.g., corners, unique patterns).
2.  **Perspective Transformation**: A 4x4 homography matrix is calculated using the four corners of the tennis court. This matrix transforms pixel coordinates `(x, y)` to real-world coordinates `(X, Y)`.
3.  **Adjusted Positions**: Each player's pixel position is first adjusted for the camera movement and then transformed using the homography matrix to get their position in meters on the court.

<div align="center">
<img src="https://i.imgur.com/37V0j9o.png" alt="Visual representation of perspective transformation on the court" width="600">
</div>

<br>

## 📁 Project Structure

```
.
├── .env                              # Environment variables (e.g., GitHub repo link)
├── .gitignore                        # Git ignore patterns
├── Football-Vision-Stats.ipynb       # Main Jupyter notebook with all analysis steps
├── runs/                             # Directory for outputs
│   ├── cropped_players/              # Cropped player images for color analysis
│   └── output_videos/                # Annotated video outputs and tracking data
│       ├── output.avi
│       ├── output_tracker_team_color_clustering.avi
│       ├── output_tracker_annotations.avi
│       ├── 08fd33_0_tracks.pkl         # Saved tracking data
│       └── 08fd33_camera_movement_stub.pkl # Saved camera movement data
├── Trained_Models/                   # Pre-trained YOLO model
│   └── best.pt
└── video_files/                      # Directory for input videos (create and add your own)
    └── 08fd33_0.mp4
```

-----

## 🤝 Contributing

We welcome contributions to this project! Here's how you can help:

1.  **Fork the repository**.
2.  **Create a new branch**: `git checkout -b feature/your-feature-name`.
3.  **Make your changes** and commit them with a clear, descriptive message.
4.  **Push to your fork**: `git push origin feature/your-feature-name`.
5.  **Create a Pull Request** to the `main` branch of this repository.

Contribution Ideas:

  - Add a user interface (GUI) to simplify video selection and analysis.
  - Implement new algorithms for player position prediction.
  - Integrate a feature to analyze passing patterns and game strategy.
  - Improve the accuracy of the ball detection and tracking.

-----

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

-----

## 📞 Contact

  - **Project Maintainer**: Your Name
  - **Project Link**: [https://github.com/lankdaniel14/Football-Vision-Stats-ML](https://github.com/lankdaniel14/Football-Vision-Stats-ML)

-----

## 🙏 Acknowledgments

  - **ultralytics**: For the powerful and easy-to-use YOLO models.
  - **supervision**: For their excellent computer vision utility library.
  - **OpenCV**: The foundation for all video and image processing in this project.
  - **Jupyter**: The environment that makes this project easy to run and share.

<div align="center">

<br>
<a href="#top">⬆ Back to Top</a>
<br>

</div>
