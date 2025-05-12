# CompanionCare: Habitat-Sim with YOLO Integration

This project integrates **Habitat-Sim** with **YOLO** object detection to create a prototype for CompanionCare, a robot vision and navigation system that can assist with Activities of Daily Living (ADLs).

## 🧭 Overview

The system combines high-fidelity indoor environment simulation from Habitat-Sim with real-time object detection using YOLO to:

1. Detect household objects relevant to ADLs (medications, furniture, food items, etc.)
2. Build and maintain a semantic map of the environment
3. Plan and execute navigation to objects of interest
4. Demonstrate assistance with common ADL tasks

## 🧩 Components

| Component | Tool | Role |
|-----------|------|------|
| Environment Simulation | **Habitat-Sim** | Simulates a realistic home with physics, RGB-D sensors |
| Object Detection | **YOLOv5/v7** | Runs on RGB camera frames to identify ADL-relevant objects |
| Navigation & Mapping | **Habitat API** | Agent movement, waypoint generation, avoidance |
| Semantic Mapping | **Custom** | Builds and maintains a map of detected objects |

## 🧱 System Architecture

```
Habitat Environment
    ├── Agent (with camera sensor)
    │     ├── RGB frame
    │     └── Depth frame
    └── Ground truth maps + object layouts

↓ frame

YOLOv7 Object Detection
    ├── Run on RGB frame
    └── Detect ["meds", "chair", "bed", "sink", etc.]

↓ detections

Semantic Map / Logic
    ├── Update known object locations
    └── Inform navigation decisions

↓ goals

Habitat Navigation Policy
    ├── Plan route to goal
    └── Avoid obstacles
```

## 🛠️ Installation

### Prerequisites

1. Python 3.7+ with pip
2. CUDA-compatible GPU (recommended for YOLO)
3. OpenCV with GUI support (for visualizations)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/companioncare-habitat-yolo.git
cd companioncare-habitat-yolo

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Habitat-Sim (see instructions for your platform)
# https://github.com/facebookresearch/habitat-sim#installation

# Download test scenes (if needed)
python -m habitat_sim.utils.datasets_download --datasets habitat_test_scenes
```

## 📊 Usage

### Basic Demo

```bash
python main.py --duration 120 --explore_time 30
```

### Custom Configuration

```bash
python main.py \
  --scene data/scene_datasets/habitat-test-scenes/apartment_1.glb \
  --yolo_weights yolov5s.pt \
  --target_objects chair,bottle,bed,bowl \
  --output_dir ./my_outputs \
  --duration 180 \
  --explore_time 60
```

### Command-Line Options

- `--scene`: Path to Habitat scene file (.glb)
- `--yolo_weights`: Path to YOLO weights file (.pt for PyTorch or .weights for Darknet)
- `--yolo_config`: Path to YOLO config file (for Darknet models)
- `--output_dir`: Directory to save outputs
- `--target_objects`: Comma-separated list of target object classes
- `--device`: Device for YOLO inference ("cuda" or "cpu")
- `--duration`: Total demo duration in seconds
- `--explore_time`: Initial exploration time in seconds
- `--no_visualization`: Disable visualization

## 📁 Code Structure

- `main.py`: Main script that integrates all components
- `environment.py`: Handles Habitat environment setup and sensor management
- `yolo_detector.py`: Interfaces with YOLO models for object detection
- `semantic_mapper.py`: Builds and maintains a semantic map of detected objects
- `navigation.py`: Plans and executes navigation based on semantic map

## 🎬 Demo Features

The demo will:

1. Initialize the Habitat environment and YOLO detector
2. Explore the environment to build a semantic map
3. Identify objects of interest for ADL support
4. Execute navigation tasks in priority order (e.g., medication reminder, hydration support)
5. Generate visualizations and statistics of the completed tasks

## 📊 Output

The system produces:

- Semantic map visualizations
- Task completion statistics
- Navigation path recordings
- Detection logs with timestamps and confidence scores
