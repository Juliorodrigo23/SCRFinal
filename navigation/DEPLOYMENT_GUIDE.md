# CompanionCare: Deployment & Testing Guide

This guide provides detailed instructions for setting up, running, and extending the CompanionCare Habitat-YOLO integration.

## 📦 Complete Installation Guide

### System Requirements

- **Operating System**: Linux (recommended), macOS, or Windows 10+
- **Python**: 3.7+ (3.8 recommended)
- **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- **CPU**: 4+ cores
- **RAM**: 16GB+
- **Storage**: 10GB+ free space

### Step-by-Step Installation

#### 1. Set Up Environment

```bash
# Clone repository
git clone https://github.com/yourusername/companioncare-habitat-yolo.git
cd companioncare-habitat-yolo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (adjust based on your CUDA version)
# For CUDA 11.6:
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
# For CPU only:
# pip install torch torchvision

# Install base requirements
pip install -r requirements.txt
```

#### 3. Install Habitat-Sim

The installation process varies by platform:

**Linux (Ubuntu)**:
```bash
conda install -c conda-forge habitat-sim headless
```

**macOS**:
```bash
conda install -c conda-forge habitat-sim
```

**Windows**:
```bash
# Windows requires building from source - see Habitat-Sim docs
# https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md#building-on-windows
```

**Alternative (any platform)**:
```bash
# Using pip (may not include all features)
pip install habitat-sim
```

#### 4. Download Test Data

```bash
# Download test scenes
python -m habitat_sim.utils.datasets_download --datasets habitat_test_scenes

# Verify installation
python -c "import habitat_sim; print(habitat_sim.__version__)"
```

## 🧪 Testing and Validation

### Basic Functionality Test

Run this script to verify all components are working:

```bash
# Test the environment module
python -c "from environment import HabitatEnvironment, get_available_scenes; scenes = get_available_scenes(); print(f'Available scenes: {list(scenes.keys())}')"

# Test YOLO (will download a model if not present)
python -c "from yolo_detector import YOLODetector; detector = YOLODetector(weights_path='yolov5s.pt'); print('YOLO detector initialized successfully')"
```

### Run Simple Demo

```bash
# Short demo with visualization
python main.py --duration 60 --explore_time 20

# Verify outputs were created
ls -la ./outputs/
```

### Expected Output

The demo should produce:
- Terminal output showing initialization, detection, and navigation status
- Multiple visualization windows showing:
  - RGB camera view with bounding boxes
  - Semantic map with detected objects
- Output files in `./outputs/` including:
  - Semantic map images
  - Task screenshots
  - Results JSON file

## 🔧 Customization Options

### Using Custom Scenes

```bash
# Download a scene from Matterport3D or other source
# Place in data/scene_datasets/

# Run with custom scene
python main.py --scene path/to/your/custom_scene.glb
```

### Using Custom YOLO Models

```bash
# For PyTorch YOLOv5/v7
python main.py --yolo_weights path/to/your/custom_model.pt

# For Darknet YOLO
python main.py --yolo_weights path/to/your/custom_model.weights --yolo_config path/to/your/custom_model.cfg
```

### Customizing Target Objects

```bash
# Specify target objects to detect
python main.py --target_objects chair,sofa,bottle,remote,cup,plant
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Missing Habitat Scenes

**Problem**: `No test scenes found. You may need to download them.`

**Solution**:
```bash
python -m habitat_sim.utils.datasets_download --datasets habitat_test_scenes
```

#### 2. CUDA/GPU Issues

**Problem**: CUDA device not found or out of memory errors

**Solution**:
```bash
# Force CPU mode
python main.py --device cpu

# Or check CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

#### 3. OpenCV Display Issues

**Problem**: Visualization windows don't appear or cause errors

**Solution**:
```bash
# Disable visualization
python main.py --no_visualization

# Or check OpenCV installation
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

## 🚀 Extending the System

### Adding New ADL Tasks

Modify the `run_demo` method in `main.py` to add new ADL task types:

```python
# Add a new task type
if "refrigerator" in detected_classes:
    adl_tasks.append({
        'name': 'food_reminder',
        'type': 'object',
        'target': 'refrigerator',
        'priority': 2,
        'description': 'Navigate to refrigerator for meal reminder'
    })
```

### Implementing New Navigation Behaviors

Extend the `ObjectGoalNavigator` class in `navigation.py`:

```python
def patrol_between_objects(self, object_class_a, object_class_b, iterations=3):
    """
    Patrol between two object types
    
    Args:
        object_class_a (str): First object class
        object_class_b (str): Second object class
        iterations (int): Number of patrol iterations
        
    Returns:
        bool: Success status
    """
    # Implementation here
```

### Adding Custom Metrics

Modify the `_execute_task` method in `main.py` to track additional metrics:

```python
# Add to the result dictionary
result.update({
    'energy_consumption': calculate_energy(task),
    'user_interaction_opportunities': estimate_interactions(task),
    'safety_score': evaluate_safety(task)
})
```

## 📊 Evaluation Methodology

### Performance Metrics

The system collects these key metrics:

1. **Detection Performance**:
   - Number of objects detected
   - Detection confidence
   - Detection FPS

2. **Navigation Performance**:
   - Task success rate
   - Path length vs. optimal path
   - Navigation time
   - Path efficiency (direct distance / traveled distance)

3. **System Performance**:
   - Overall FPS
   - Component timing breakdowns

### Conducting a Full Evaluation

```bash
# Run a longer evaluation
python main.py --duration 300 --explore_time 60 --output_dir ./evaluation_results

# Analyze results
python -c "import json; with open('./evaluation_results/demo_results.json') as f: results = json.load(f); print(f'Success rate: {results[\"summary\"][\"success_rate\"]:.2f}')"
```

## 📊 Visualization Options

### Real-time Visualizations

The system provides these real-time windows:

1. **RGB Camera View**: Shows agent's view with object detections
2. **Semantic Map**: Shows built map of environment with detected objects
3. **Navigation View**: Shows path planning and execution

### Generating Summary Reports

After running a demo, generate a comprehensive report:

```bash
# Install additional visualization packages
pip install plotly pandas

# Run report generation script (create this script to process results)
python generate_report.py --results_file ./outputs/demo_results.json
```

This guide should help you deploy and test your CompanionCare prototype effectively. For additional assistance, consult the documentation for [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) and [YOLO](https://github.com/ultralytics/yolov5).
