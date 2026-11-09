# EduSphere - Student Presence & Focus Analytics Pipeline

## Overview
An end-to-end Computer Vision and Machine Learning pipeline designed to automatically analyze student presence and focus levels during educational video sessions using OpenCV, MediaPipe, and Pandas.

## Pipeline Stages

### 1. Video Ingestion & Preprocessing
- Captures and processes educational video frames sequentially using OpenCV (cv2).
- Normalizes resolution and extracts face detection bounding boxes in real-time.

### 2. Feature Extraction & Focus Estimation
- Utilizes MediaPipe Face Mesh to extract facial landmarks.
- Computes 3D head rotation angles (yaw, pitch, and roll) to determine if a student is present and actively focused.

### 3. Evaluation & Metrics Logging
- Evaluates frame-by-frame session metrics and calculates total presence and focus percentages.
- Automatically exports structured performance logs and analytical reports into CSV files.

### 4. Performance Visualization
- Generates analytical comparison charts and bar graphs (plt.bar) to compare student engagement across multiple sessions using Pandas and Matplotlib.

## Requirements
- Python 3.x
- OpenCV (opencv-python)
- MediaPipe
- NumPy
- Pandas
- Matplotlib
