# Emergency Vehicle Detection for Autonomous Vehicles

A perception subsystem designed for **Autonomous Vehicles (AVs)** to detect approaching emergency vehicles using multi-modal sensor fusion (Vision + Audio).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Configuration](#configuration)
- [License](#license)

## Overview

In the context of **Level 4/5 Autonomous Driving**, safely interacting with emergency vehicles is a critical edge case. This project implements a **perception module** that monitors the AV's environment (specifically rear and surround-view camera feeds) to identify ambulances, fire trucks, and police vehicles.

Crucially, it answers two questions for the AV's planning stack:
1.  **Is there an emergency vehicle behind/around me?** (Vision)
2.  **Is it in active emergency mode (Siren ON)?** (Audio)

By fusing these signals, the system provides a high-confidence "Yield Trigger" to the vehicle's control system, enabling it to autonomously pull over or clear the path, mimicking responsible human driver behavior.

## Features

-   **Autonomous Decision Support**: Differentiates between a parked ambulance and one approaching with sirens blaring, preventing unnecessary stops.
-   **Multi-Modal Sensor Fusion**: Combines visual data from cameras with audio data from external microphones.
-   **Video-Optimized Perception**: Vision model tuned with rotation/shear augmentation to handle vibration and perspective shifts from a moving vehicle.
-   **Enhanced Audio Classification**: robust siren detection using statistical spectral features (MFCC Mean + StdDev) to filter out wind and engine noise.
-   **Real-time Inference**: Optimized pipelines suitable for running on vehicular edge compute units.

## Project Structure

```
emergency-vechile-detection/
├── configs/                 # Configuration files
├── data/                    # Dataset directory
│   ├── audio/               # Audio samples (siren/not_siren)
│   ├── train/               # Vision training data
│   ├── valid/               # Vision validation data
│   └── test/                # Vision test data
├── models/                  # Trained models storage
│   ├── audio_classifier.joblib
│   └── best.pt (YOLO)
├── predict_data/            # Input files for prediction (images/videos)
├── results/                 # Output plots and analysis
│   ├── audio/               # Audio training plots
│   ├── vision/              # Vision training plots
│   └── predictions/         # Annotated output videos/images
├── src/                     # Source code
│   ├── audio/               # Audio processing & classification
│   ├── pipeline/            # Inference pipeline & fusion logic
│   └── vision/              # YOLO detector & analysis
├── training/                # Training scripts
│   ├── modal/               # Cloud training (Modal)
│   └── train_audio.py       # Local audio training
├── train.py                 # Main unified training entrypoint
├── predict.py               # Main inference entrypoint
├── REPORT.md                # Detailed project report
└── README.md                # This file
```

## Installation

### Prerequisites
-   Python 3.10+
-   FFmpeg (for audio processing)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd emergency-vechile-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Training

The project features a unified interactive training hub.

```bash
python train.py
```
Follow the interactive prompts to select:
-   **Model**: Vision (YOLOv8s) or Audio (Random Forest).
-   **Platform**: Local Machine or Modal Cloud.

**Automated Remapping**: The training script automatically cleans and remaps the dataset from 24 noisy classes to 4 functional classes (`ambulance`, `fire_truck`, `police`, `non_emergency_vehicle`) before training begins.

### 2. Prediction

Run the interactive prediction hub to simulate the AV perception stack:

```bash
python predict.py
```

-   **Source**: Feed a video file (simulating a camera stream).
-   **Output**: The system will annotate frames and output the "Emergency Status" which would be sent to the planning module.

### 3. Command Line Interface

```bash
# Simulate processing a rear-view camera feed
python predict.py --source predict_data/rear_view.mp4 --video --save
```

## Methodology

### Vision Model
-   **Architecture**: YOLOv8 Small (`yolov8s.pt`). Selected as a robust first-principles choice for its balance of accuracy and real-time performance.
-   **Data Strategy**: The original 24 noisy classes are pre-processed and re-engineered into 4 clear functional classes: `ambulance`, `fire_truck`, `police`, and `vehicle`. This hard normalization ensures the model learns the core concepts without ambiguity.
-   **Augmentation**: Enhanced with **Rotation (±5°)** and **Shear (2.5°)** to simulate real-world camera motion and perspective changes, improving robustness for autonomous vehicle applications.

### Audio Model
-   **Architecture**: Custom Convolutional Neural Network (CNN) designed for spectrogram analysis.
-   **Features**: Log-Mel Spectrograms, which are 2D image-like representations of sound (frequency over time). This allows the CNN to learn time-varying patterns unique to sirens.
-   **Task**: Binary Classification ("siren" vs "not_siren"). The CNN learns directly from the visual patterns in the spectrograms.

## Results

Detailed results and analysis plots can be found in the `results/` directory after training.

**Audio Analysis (`results/audio/`)**:
-   `learning_curve.png`: Shows model loss and accuracy over epochs.
-   `confusion_matrix.png`: Classification accuracy breakdown.
-   `prediction_confidence.png`: Histogram of model certainty for siren detection.
-   `feature_distribution.png`: PCA visualization of the learned feature space (embeddings).
-   `average_spectrograms.png`: Visual "feature importance" showing the average sound patterns for Siren vs. Noise.

**Vision Analysis (`results/vision/`)**:
-   `results.png`: Master dashboard of training metrics (Box Loss, Class Loss, mAP).
-   `confusion_matrix.png`: Shows inter-class confusions for object detection.
-   `BoxPR_curve.png`: Precision-Recall curve, indicating the model's detection performance trade-offs.
-   `val_batchX_pred.jpg`: Actual bounding box predictions on validation images (Visual Proof).
-   `val_batchX_labels.jpg`: Ground truth labels for validation images, for comparison.

For a full academic report, please refer to [REPORT.md](REPORT.md).
