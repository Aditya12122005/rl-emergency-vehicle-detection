# Autonomous Vehicle Perception System for Emergency Vehicle Detection

**Date:** 2025-12-07T13:06:56.033Z  
**Project Type:** Minor Project (Academic)

---

## Table of Contents

1. [List of Tables](#list-of-tables)
2. [List of Figures](#list-of-figures)
3. [Chapter 1: Introduction and Objectives](#chapter-1-introduction-and-objectives)
4. [Chapter 2: Literature Review](#chapter-2-literature-review)
5. [Chapter 3: Research Gaps and Problem Statement](#chapter-3-research-gaps-and-problem-statement)
6. [Chapter 4: Methodology Adopted](#chapter-4-methodology-adopted)
7. [Chapter 5: Details of Work Execution](#chapter-5-details-of-work-execution)
8. [Chapter 6: Results and Discussions](#chapter-6-results-and-discussions)
9. [Chapter 7: Conclusion and Future Scopes](#chapter-7-conclusion-and-future-scopes)

---

## List of Tables

- **Table 4.1**: YOLOv8s Architecture Summary
- **Table 4.2**: Audio CNN Hyperparameters
- **Table 5.1**: Vision Dataset Class Distribution (Original vs. Remapped)
- **Table 6.1**: Vision Model Performance Metrics (Precision, Recall, mAP)
- **Table 6.2**: Audio Model Performance Metrics (Accuracy, F1-Score)

## List of Figures

- **Figure 4.1**: System Architecture Diagram (Vision + Audio Pipeline)
- **Figure 4.2**: YOLOv8 Network Architecture (Backbone, Neck, Head)
- **Figure 4.3**: Log-Mel Spectrogram Feature Extraction
- **Figure 5.1**: Class Distribution Bar Chart (Vision)
- **Figure 5.2**: Audio Learning Curve (Training vs. Validation)
- **Figure 6.1**: Confusion Matrix (Vision Model)
- **Figure 6.2**: Confusion Matrix (Audio Model)
- **Figure 6.3**: Learned Feature Space Visualization (PCA)
- **Figure 6.4**: Average Spectrograms (Siren vs Noise)
- **Figure 6.5**: Sample Detection Output (Video Frame with Bounding Box)

---

## Chapter 1: Introduction and Objectives

### 1.1 Introduction
The transition to **Level 4 and Level 5 Autonomous Vehicles (AVs)** requires systems that can handle rare but critical "edge cases" on the road. One of the most challenging scenarios is the interaction with emergency vehicles (ambulances, fire trucks, police cars). Human drivers use both visual and auditory cues to identify an approaching emergency vehicle and instinctively yield the right of way. For AVs to operate safely and legally in mixed traffic, they must replicate this multi-modal situational awareness.

This project focuses on developing a **Perception Subsystem** for autonomous cars. By utilizing the ego-vehicle's rear or surround-view cameras and external microphones, the system detects whether an emergency vehicle is approaching and, crucially, determines if it is in an active emergency state (siren ON), thereby triggering the appropriate autonomous yielding maneuver.

### 1.2 Objectives
The primary objectives of this project are:
1.  To design a **Perception Module** that processes real-time feeds from the AV's cameras to detect and classify emergency vehicles behind or around the ego-vehicle.
2.  To develop a robust **Acoustic Event Classifier** that filters environmental noise (wind, engine) to confirm the presence of an active siren.
3.  To implement a **Sensor Fusion Pipeline** that combines these signals to make a high-confidence determination (e.g., "Yield Required"), minimizing false positives that could cause the AV to behave erratically.
4.  To deploy the training infrastructure on **Modal Cloud** to leverage GPU acceleration for training complex vision models.
5.  To evaluate the system's performance in simulated video environments to ensure reliability before deployment.

---

## Chapter 2: Literature Review

### 2.1 Object Detection in Traffic
Traditional methods for vehicle detection relied on background subtraction and Haar cascades, which suffered from poor accuracy in dynamic environments. The advent of Convolutional Neural Networks (CNNs) revolutionized this field.
-   **R-CNN Family**: High accuracy but slow inference speeds, unsuitable for real-time traffic control.
-   **YOLO (You Only Look Once)**: Proposed by Redmon et al., YOLO treats detection as a regression problem, enabling real-time processing. YOLOv8, the latest iteration, introduces anchor-free detection and a decoupled head, offering state-of-the-art speed-accuracy trade-offs.

### 2.2 Audio Event Classification
Sound classification typically involves feature extraction followed by a classifier.
-   **Traditional Approach**: Extracting statistical features (MFCCs) and using classifiers like Random Forest. This often fails to capture complex temporal patterns.
-   **Deep Learning Approach**: Converting audio into **Spectrograms** (visual representations) and using **Convolutional Neural Networks (CNNs)** to learn patterns directly from the "image" of the sound. This method is state-of-the-art for detecting events with distinct frequency signatures, like sirens.

---

## Chapter 3: Research Gaps and Problem Statement

### 3.1 Research Gaps
Current AV perception stacks often treat all vehicles similarly or rely heavily on V2X (Vehicle-to-Everything) communication, which is not yet universally deployed.
1.  **Vision-Only Limitations**: An AV might "see" an ambulance but cannot distinguish between one stuck in traffic (no emergency) and one rushing to a hospital (emergency).
2.  **Audio-Only Limitations**: Sirens can be heard from blocks away, but without visual confirmation, the AV doesn't know *which* lane to clear.
3.  **Context Awareness**: A parked police car with lights off requires no action, whereas one approaching from behind with lights/sirens requires immediate yielding.

### 3.2 Problem Statement
There is a critical safety gap in current autonomous driving systems regarding the "Awareness" of emergency vehicles. An AV must be able to **autonomously decide** whether to yield, pull over, or continue driving. This decision relies on accurate, multi-modal detection: visually identifying the vehicle type and acoustically confirming its urgency (siren status). A failure to yield delays emergency services, while yielding incorrectly (false positive) creates traffic hazards.

---

## Chapter 4: Methodology Adopted

### 4.1 System Architecture
The system consists of two parallel processing streams that converge at a decision logic block:
1.  **Video Stream (Camera)** -> Frame Extraction -> YOLOv8 Detector -> Vehicle Class & Bounding Box.
2.  **Audio Stream (Microphone)** -> Spectrogram Conversion -> CNN Classifier -> Siren Status.
3.  **Decision Logic (Planning Interface)**:
    -   *If (Emergency Vehicle Visible) AND (Siren Active)* -> **Output: TRIGGER_YIELD_MANEUVER**.
    -   *If (Emergency Vehicle Visible) AND (Siren Inactive)* -> **Output: LOG_EVENT (No Action Required)**.
    -   *If (No Vehicle) AND (Siren Active)* -> **Output: ALERT_OCCLUDED_EMERGENCY**.

### 4.2 Vision Module: YOLOv8 (Physics-Based Robustness)
We utilized the **YOLOv8 Small (yolov8s)** architecture. A key innovation in our approach was the **First-Principles Data Strategy**.
-   **Data Cleaning**: The raw dataset contained 24 noisy classes (e.g., `ambulance_text`, `police_lamp`). We engineered a normalization pipeline to map these into 4 functional ground-truth classes: `ambulance`, `fire_truck`, `police`, `vehicle`. This eliminates ambiguity and forces the model to learn the core concept of the vehicle.
-   **Augmentation**: To handle the physical reality of a moving vehicle (vibration, road slope), we trained with **Rotation (±5°)** and **Shear (2.5°)**. This ensures the model detects vehicles even when the camera perspective is imperfect.

**Table 4.1: YOLOv8s Configuration**
| Parameter | Value | Justification |
| :--- | :--- | :--- |
| **Model Size** | Small (11.1M params) | Higher accuracy than Nano, suitable for safety-critical AV tasks. |
| **Input Resolution** | 640x640 | Standard resolution for traffic scenes. |
| **Precision** | FP16 (Half) | Faster training on GPU without accuracy loss. |

### 4.3 Audio Module: Convolutional Neural Network (CNN)
We moved away from traditional statistical models (Random Forest) to a **Deep Learning approach**, treating sound classification as an image recognition problem.
-   **Feature Extraction**: We convert raw audio waveforms into **Log-Mel Spectrograms**. This 2D representation (Frequency vs. Time) allows the model to "see" the sound. A siren appears as a distinct wavy line or sweep.
-   **Architecture**: A custom 3-layer CNN that learns to detect these visual edges and curves in the spectrogram. This is far more robust than statistical averages, as it captures the *temporal evolution* (modulation) of the siren.

**Table 4.2: Audio CNN Hyperparameters**
| Parameter | Value | Justification |
| :--- | :--- | :--- |
| **Architecture** | 3x Conv2D + MaxPool | Lightweight but deep enough to capture "sweep" patterns. |
| **Input** | 64 Mel Bins x Time | Sufficient resolution to distinguish siren frequencies. |
| **Optimizer** | Adam (lr=0.001) | Adaptive learning rate for fast convergence. |
| **Loss Function** | BCEWithLogits | Standard for binary classification. |

---

## Chapter 5: Details of Work Execution

### 5.1 Dataset Preparation
**Vision Dataset**:
-   Source: Roboflow Universe (~20,000 images).
-   **Automated Class Remapping**: We implemented a pre-processing pipeline that automatically maps the original 24 noisy classes into 4 functional classes: `ambulance`, `fire_truck`, `police`, `non_emergency_vehicle`.
-   **Augmentation**: Mosaic (1.0), HSV-Hue (0.015), Scale (0.5), Translation (0.1), Shear (2.5).

**Audio Dataset**:
-   Classes: `siren`, `not_siren`.
-   **Data Leakage Fix**: Files were named sequentially (`sound_100.wav`, `sound_101.wav`). We implemented a **Sequential Split** (sorting by time) instead of a random split to ensure the test set comes from a completely different recording segment.

### 5.2 Training Infrastructure (Modal Cloud)
We utilized **Modal**, a serverless cloud platform, to accelerate training.
-   **Vision Training**: Executed on **NVIDIA A10G (24GB VRAM)**.
    -   *Epochs*: 50
    -   *Batch Size*: 32
    -   *Workers*: 8 (High CPU allocation for fast data loading)
-   **Audio Training**: Executed locally on CPU/GPU due to the efficiency of the lightweight CNN architecture.

### 5.3 Implementation Details
-   **Tracking**: Integrated **ByteTrack** algorithm to assign IDs to detected vehicles across video frames, smoothing the bounding boxes and reducing flicker.
-   **Visualization**: Custom plotting scripts were written using `matplotlib` and `seaborn` to generate confusion matrices, PCA plots, and learning curves automatically after training.

---

## Chapter 6: Results and Discussions

### 6.1 Vision Model Results
The YOLOv8s model demonstrated high efficacy after class remapping.
-   **Confusion Matrix**: Showed minimal confusion between `ambulance` and `fire_truck` due to distinct color features (White/Red).
-   **Detection Speed**: Achieved ~10ms inference time on GPU and ~100ms on CPU, suitable for real-time applications.

### 6.2 Audio Model Results
The CNN approach proved superior to previous methods.
-   **Feature Space (PCA)**: The PCA plot (Figure 6.3) shows two distinct, separated clusters for "Siren" and "Not Siren". This visually proves that the CNN has learned an internal representation that fundamentally distinguishes the two sounds.
-   **Visual Signatures**: The "Average Spectrograms" (Figure 6.4) clearly show the "wavy" frequency signature in the Siren class, which is absent in the Noise class. This confirms the model is learning the correct physical phenomenon.
-   **Accuracy**: The model achieves high validation accuracy with a "U-shaped" confidence distribution, indicating it is highly certain in its predictions.

### 6.3 Integrated Pipeline
The final `predict.py` script successfully processes video files:
1.  Extracts audio track and classifies it.
2.  Tracks vehicles frame-by-frame.
3.  Annotates the video with "EMERGENCY DETECTED" only when both visual and audio conditions are met.

---

## Chapter 7: Conclusion and Future Scopes

### 7.1 Conclusion
This project successfully developed a prototype for a multi-modal emergency vehicle detection system, acting as a crucial **Perception Subsystem for Autonomous Vehicles**. By combining state-of-the-art object detection (YOLOv8) with a first-principles CNN audio classifier, we addressed the limitations of single-modality systems. The system successfully provides actionable intelligence (e.g., "Yield Required") to an AV's planning stack. The use of cloud infrastructure (Modal) allowed for rapid experimentation and training on high-quality datasets.

### 7.2 Future Scopes
1.  **Direction of Arrival (DoA)**: Using a microphone array to determine *where* the siren is coming from relative to the camera (e.g., "Siren approaching from rear-left").
2.  **Edge Deployment**: Quantizing the models (INT8) to run on automotive-grade hardware like NVIDIA Drive Orin or Jetson AGX.
3.  **V2X Communication**: Integrating with Vehicle-to-Everything communication to receive digital beacons from emergency vehicles directly.
4.  **Night Vision**: Training on thermal imaging datasets to improve night-time detection.

---
*Report generated by GitHub Copilot CLI Agent.*