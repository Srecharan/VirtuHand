# VirtuHand: Real-time Hand Gesture Recognition for AR Interaction

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8-blue.svg)
![Unity](https://img.shields.io/badge/unity-2022.3.5f1-green.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.13.1-orange.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.9.0-red.svg)

## Overview

VirtuHand is a sophisticated real-time hand gesture recognition system implementing a hybrid architecture that combines classical computer vision with deep learning approaches. The system leverages depth sensing camera's capabilities enhanced by Extended Kalman filtering for precise 3D tracking, while incorporating both MediaPipe-based gesture recognition and experimental ONNX neural network implementations for robust hand detection and pose estimation.

<div align="center">
  <a href="assets/VirtuHand_overview.pdf">
    <img src="assets/First_page.png" width="600" alt="VirtuHand Technical Overview"/>
    <p><i>Click on the image to view the complete Technical Overview PDF</i></p>
  </a>
</div>

### Interactive Demo: Virtual Flower Arrangement
![Full Demo](assets/full_demo_gesture.gif)

*📌 For full quality, watch the video on [YouTube](https://youtu.be/eRFWZjJbcgI)*

### Hand Articulation & Rigging Demo
![Hand Rigging](assets/hand_rig.gif)


## Technical Architecture

### System Architecture

<div align="center">
  <img src="assets/virtuhand_sys.png" width="800"/>
  <p><i>Complete system architecture showing data flow from camera through Python backend to Unity frontend</i></p>
</div>

### Unity Integration
- **Hand Model**: Fully articulated hand model with inverse kinematics
- **Real-time Physics**: Dynamic object interaction and collision detection
- **Custom Shaders**: Advanced material rendering and effects
- **WebSocket Communication**: Low-latency data streaming protocol

## Core Features

### Advanced Hand Tracking & Depth Sensing
- **Depth Camera Integration**:
  - Depth sensing camera (30 FPS, 640x480)
  - Multi-stage depth filtering pipeline
  - Sub-millimeter precision in optimal conditions
  - Custom depth data preprocessing and normalization

- **Extended Kalman Filter Implementation**:
  - 2D state vector (position, velocity) estimation
  - Optimized measurement and process noise matrices
  - Dynamic time-step handling (30Hz update rate)
  - Custom covariance tuning for hand motion characteristics
  - Advanced outlier rejection for robust tracking

- **3D Position Tracking**:
  - MediaPipe hand landmark detection (21 keypoints)
  - Palm center calculation using weighted landmark averaging
  - Geometric handedness determination using cross-product analysis
  - Quaternion-based rotation tracking with Kalman smoothing
  - Real-time coordinate space transformation

### Static Gesture Recognition System
- **Geometric Analysis Pipeline**:
  - Joint angle calculation using landmark triplets
  - Adaptive finger state detection with angle thresholds
  - Palm orientation analysis using normal vectors
  - Real-time confidence scoring system

- **Supported Gestures**:
  - GRAB: Finger curl analysis with palm-relative distances
  - OPEN_PALM: Extended finger validation with strict thresholds
  - PINCH: Precision thumb-index distance monitoring
  - POINT: Index extension with others closed detection

  <div align="center">
  <img src="assets/static_gestures.gif" width="700"/>
  <p><i>Demonstration of supported static gestures: GRAB, OPEN_PALM, PINCH, and POINT</i></p>
</div>

### Dynamic Gesture Recognition System
- **Dynamic Gesture Recognition Architecture**:

<div align="center">
  <img src="assets/GRU.png" width="600"/>
  <p><i>GRU-based dynamic gesture recognition pipeline with sequence preprocessing and temporal smoothing</i></p>
</div>

- **Custom Training Pipeline**:
  - Dataset: 20 sequences per gesture, 30 frames each
  - GRU architecture (input:63, hidden:32, layers:2)
  - Real-time sequence buffer management
  - Custom data augmentation for rotation invariance

- **Motion Pattern Analysis**:
  - Velocity component extraction (dx, dy)
  - Horizontal/vertical motion ratio analysis
  - Specialized pattern detectors:
    - SWIPE: Direction and magnitude validation
    - CIRCLE: Rotation and shape consistency checking
    - WAVE: Peak-valley detection with amplitude analysis

- **Performance Metrics**:
  - Recognition latency: <33ms
  - Gesture confidence threshold: 0.6
  - Minimum sequence length: 30 frames
  - Real-time smoothing with 5-frame minimum consistency

<div align="center">
  <img src="assets/dynamic_gestures.gif" width="700"/>
  <p><i>Demonstration of supported dynamic gestures: SWIPE, CIRCLE, and WAVE</i></p>
</div>

### Neural Network Integration 
- **ONNX Neural Network Architecture**:
<div align="center">
  <img src="assets/ONNX.png" width="700"/>
  <p><i>ONNX neural network pipeline with parallel palm detection and hand landmark models</i></p>
</div>

- **ONNX-Based Hand Detection**:
  - Attempted replacement of MediaPipe's hand detection pipeline with ONNX models
  - Two-stage detection system:
    - Palm Detection (192x192 input)
    - Hand Landmark Detection (224x224 input)
  - Unity Barracuda engine integration for GPU acceleration
  - Custom tensor preprocessing pipeline
  - FP16 quantization for model optimization
- **Performance Targets**:
  - Palm Detection: 8-10ms inference time
  - Hand Landmark: 12-15ms inference time
  - Overall pipeline latency: < 33ms (30+ FPS)
  - Memory footprint: ~150MB during runtime

### Real-time Communication System
- **WebSocket Protocol**:
  - Asynchronous bidirectional communication
  - Custom JSON message protocol for gesture data
  - Optimized packet size (< 1KB per frame)
  - Auto-reconnection with exponential backoff
  - Ping-pong disabled for reduced latency
- **Data Flow**:
  - Hand position/rotation (30Hz)
  - Gesture classifications (real-time)
  - Joint angles for hand model
  - Dynamic gesture events

## Installation and Usage

1. **Clone the Repository**
```bash
git clone https://github.com/Srecharan/VirtuHand.git
cd VirtuHand
```

2. **Unity Setup**
   - Download and install Unity 2022.3.5f1
   - Import required packages from Package Manager:
     - Native WebSocket
     - Barracuda
     - Intel RealSense SDK
   - Choose one of the following options:
     - Open the existing HandGestureTest2 scene from the project
     - Create a new scene and import objects from project assets
     - Create a new scene and import assets from Unity Asset Store

3. **Run the Backend**
```bash
# Start the Python server
python tests/test_server.py
```
After running the server, switch to Unity and enter Game mode to begin hand tracking.

**Important Notes:**
- Hand Model Setup:
  - Ensure proper rigging of all finger joints (21 points)
  - Each joint should have proper rotation constraints
  - Thumb requires special attention for natural movement
  - Configure inverse kinematics for realistic hand movement
- Script Attachment:
  - Attach HandGestureController.cs to the hand model
  - Configure WebSocket connection parameters
  - Set up gesture detection thresholds
- Camera Setup:
  - Position the depth sensing camera at appropriate height
  - Ensure proper lighting for optimal tracking

## Acknowledgments
- [MediaPipe Hand Landmark Detection](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) by Google
- Unity Technologies and Unity Asset Store
- Intel RealSense SDK
- Unity ML-Agents

## License
This project is licensed under the MIT License - see the LICENSE file for details