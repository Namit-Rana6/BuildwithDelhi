# YOLOv8 Object Detection: FireExtinguisher, ToolBox & OxygenTank Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Created by Team Oink Boink** 🐷

A comprehensive YOLOv8 object detection model for identifying three critical safety equipment classes: **FireExtinguisher**, **ToolBox**, and **OxygenTank**. This project demonstrates the complete machine learning workflow from data preparation to model deployment.

## 🎯 Project Overview

This repository contains a custom-trained YOLOv8x model that achieves high accuracy in detecting safety equipment in industrial environments. The model is trained on a curated dataset and optimized for real-world deployment scenarios.

### Key Features

- **High Accuracy**: Achieves 93.1% precision and 83.2% recall on test data
- **Outstanding Training Performance**: Reached 97.70% mAP50 during validation
- **Three Class Detection**: FireExtinguisher, ToolBox, OxygenTank
- **Robust Training**: 100 epochs with advanced augmentation techniques
- **Production Ready**: Optimized hyperparameters and early stopping
- **Comprehensive Evaluation**: Detailed metrics and confusion matrix analysis
- **Complete Colab Notebook**: Full training pipeline included for easy reproduction

## 📊 Model Performance

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| FireExtinguisher | 91.2% | 83.1% | 87.4% | 79.3% |
| ToolBox | 95.3% | 84.4% | 91.8% | 88.1% |
| OxygenTank | 92.6% | 82.2% | 90.1% | 83.7% |
| **Overall** | **93.1%** | **83.2%** | **89.8%** | **83.7%** |

### Key Metrics
- **Overall mAP50**: 89.8% - Excellent detection accuracy
- **Overall mAP50-95**: 83.7% - Strong performance across IoU thresholds
- **Inference Speed**: ~73.6ms per image
- **Model Size**: YOLOv8x architecture for maximum accuracy

## 🛠️ Installation & Setup

### Prerequisites

```bash
Python 3.8+
CUDA-capable GPU (recommended)
```

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/your-username/yolov8-safety-detection.git
cd yolov8-safety-detection
```

2. **Install dependencies**
```bash
pip install ultralytics opencv-python pillow matplotlib seaborn pandas
```

3. **Download the trained model**
   - Download `best.pt` from the releases section
   - Place it in the project root directory

4. **Download the dataset (Required for training/validation)**
   - Get the HackByte Dataset from: [https://falcon.duality.ai/secure/documentation/hackathon&utm_source=hackathon&utm_medium=instructions&utm_campaign=sunhacks](https://falcon.duality.ai/secure/documentation/hackathon&utm_source=hackathon&utm_medium=instructions&utm_campaign=sunhacks)
   - The dataset contains training, validation, and test images with YOLO format annotations
   - Required for model training, validation, or custom evaluation

## 🚀 Usage

### Using the Pre-trained Model (Inference Only)

If you just want to use our trained model for inference without training:

```python
from ultralytics import YOLO
from PIL import Image

# Load the trained model
model = YOLO('best.pt')

# Run inference on an image
results = model('path/to/your/image.jpg')

# Display results
results[0].show()
```

### Complete Training Pipeline in Google Colab

Our model was trained using Google Colab with Google Drive integration. **We've included the complete Colab notebook in this repository** - simply mount your Google Drive, add the dataset zip file, and run the cells to reproduce our results!

#### Quick Start with Our Colab Notebook

1. **Download the included notebook**: `training_notebook.ipynb` from this repository
2. **Open in Google Colab**: Upload the notebook to Google Colab
3. **Mount Google Drive**: Run the first cell to mount your drive
4. **Add dataset**: Upload `Hackathon_Dataset.zip` to your Google Drive root
5. **Run all cells**: The notebook will automatically handle extraction, training, and evaluation

The notebook includes all necessary code for:
- Automatic dataset extraction and preparation
- YAML configuration file generation  
- Model training with optimized hyperparameters
- Performance evaluation and visualization
- Model download for deployment

#### Manual Setup (Step-by-Step Guide)

For those who want to understand each step, here's the complete training process:

#### Step 1: Setup Google Colab Environment

1. **Open Google Colab** and create a new notebook
2. **Mount Google Drive** to access your dataset:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. **Upload the dataset** to your Google Drive:
   - Download the HackByte Dataset from the provided link
   - Upload `Hackathon_Dataset.zip` to your Google Drive root (`/content/drive/MyDrive/`)

#### Step 2: Extract and Prepare Dataset

```python
import zipfile
import os

# Define paths
zip_file_path = '/content/drive/MyDrive/Hackathon_Dataset.zip'
extract_dir = '/tmp/hackathon_dataset'

# Create extraction directory
os.makedirs(extract_dir, exist_ok=True)

# Extract the dataset
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Dataset extracted to: {extract_dir}")
```

#### Step 3: Create YOLO Configuration File

```python
# Create YAML configuration for YOLO training
yaml_content = """
path: /tmp/hackathon_dataset/HackByte_Dataset/data

train: train/images
val: val/images
test: test/images

nc: 3
names: ['FireExtinguisher', 'ToolBox', 'OxygenTank']
"""

# Save the configuration file
with open('/tmp/hackathon_dataset/HackByte_Dataset/yolo_params.yaml', 'w') as f:
    f.write(yaml_content)
```

#### Step 4: Install Dependencies and Train

```python
# Install YOLOv8
!pip install -q ultralytics

# Import and initialize YOLO
from ultralytics import YOLO

# Load pre-trained YOLOv8x model
model = YOLO("yolov8x.pt")

# Start training with optimized hyperparameters
model.train(
    data="/tmp/hackathon_dataset/HackByte_Dataset/yolo_params.yaml",
    epochs=100,
    imgsz=896,
    batch=8,
    optimizer="AdamW",
    lr0=1e-4,
    weight_decay=0.001,
    patience=15,
    dropout=0.10,
    label_smoothing=0.05,
    
    # Data augmentation settings
    mosaic=0.4,
    mixup=0.15,
    hsv_h=0.015,
    hsv_s=0.6,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    
    project="hackbyte-final",
    name="yolov8x-best",
    save=True,
    val=True
)
```

#### Step 5: Download Trained Model

```python
from google.colab import files

# Download the best model weights
best_model_path = "hackbyte-final/yolov8x-best/weights/best.pt"
files.download(best_model_path)
```

### Batch Processing

```python
import os
from ultralytics import YOLO

model = YOLO('best.pt')

# Process multiple images
image_folder = 'path/to/images/'
for filename in os.listdir(image_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        results = model(os.path.join(image_folder, filename))
        results[0].save(f'outputs/{filename}')
```

### Model Validation

```python
# Validate on your dataset
metrics = model.val(data='path/to/your/dataset.yaml')
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

## 📁 Repository Structure

```
yolov8-safety-detection/
├── README.md                    # This comprehensive guide
├── requirements.txt             # Python dependencies
├── best.pt                      # Pre-trained YOLOv8 model weights
├── training_notebook.ipynb      # Complete Google Colab training pipeline
└── data/                        # Dataset structure (download from provided link)
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

## 📊 Dataset Information

### Dataset Source
- **Download Link**: [https://falcon.duality.ai/secure/documentation/hackathon&utm_source=hackathon&utm_medium=instructions&utm_campaign=sunhacks](https://falcon.duality.ai/secure/documentation/hackathon&utm_source=hackathon&utm_medium=instructions&utm_campaign=sunhacks)
- **Name**: HackByte Dataset (Hackathon_Dataset.zip)
- **Size**: Contains training, validation, and test splits
- **Format**: Images with YOLO format bounding box annotations

### Dataset Format
- **Format**: YOLO format with bounding box annotations
- **Image Resolution**: Optimized for 896x896 input size
- **Split**: Train/Val/Test with proper stratification

### Data Configuration (yolo_params.yaml)
```yaml
# Configuration file for YOLO training
path: /tmp/hackathon_dataset/HackByte_Dataset/data  # Dataset root path
train: train/images                                  # Training images path
val: val/images                                      # Validation images path
test: test/images                                    # Test images path
nc: 3                                               # Number of classes
names: ['FireExtinguisher', 'ToolBox', 'OxygenTank'] # Class names
```

## 🔧 Training Configuration

### Google Colab Training Environment
Our model was trained using the following setup:
- **Platform**: Google Colab (Free/Pro)
- **GPU**: NVIDIA T4 (15GB VRAM)
- **Runtime**: Python 3.8+ with CUDA support
- **Storage**: Google Drive for dataset storage and mounting

### Training Process Overview

1. **Data Preparation**:
   - Dataset uploaded to Google Drive as `Hackathon_Dataset.zip`
   - Mounted Google Drive to `/content/drive`
   - Extracted dataset to `/tmp/hackathon_dataset`
   - Generated YOLO configuration file automatically

2. **Model Initialization**:
   - Started with pre-trained YOLOv8x weights (`yolov8x.pt`)
   - YOLOv8x chosen for maximum accuracy (largest model variant)

3. **Hyperparameter Optimization**:
   The model was trained with carefully optimized hyperparameters:

```python
model.train(
    data="/tmp/hackathon_dataset/HackByte_Dataset/yolo_params.yaml",
    epochs=100,                   # Extended training for thorough learning
    imgsz=896,                    # High resolution for better small object detection
    batch=8,                      # Optimized for T4 GPU memory (15GB)
    optimizer="AdamW",            # Superior to SGD for this dataset size
    lr0=1e-4,                     # Conservative learning rate for stability
    weight_decay=0.001,           # L2 regularization to prevent overfitting
    patience=15,                  # Early stopping after 15 epochs without improvement
    dropout=0.10,                 # Dropout regularization for generalization
    label_smoothing=0.05,         # Reduces overconfidence on training labels
    
    # Advanced data augmentation pipeline
    mosaic=0.4,                   # 40% mosaic augmentation (4 images combined)
    mixup=0.15,                   # 15% mixup augmentation (image blending)
    hsv_h=0.015,                  # Slight hue variation (±1.5%)
    hsv_s=0.6,                    # Saturation changes (±60%)
    hsv_v=0.4,                    # Value/brightness changes (±40%)
    translate=0.1,                # Random translation (±10%)
    scale=0.5,                    # Random scaling (±50%)
    
    # Project organization
    project="hackbyte-final",     # Main project folder
    name="yolov8x-best",         # Experiment name
    save=True,                    # Save checkpoints during training
    val=True                      # Run validation after each epoch
)
```

### Why These Hyperparameters?

- **High Image Size (896px)**: Better detection of small objects and fine details
- **Moderate Batch Size (8)**: Balances training speed with GPU memory constraints
- **AdamW Optimizer**: Better convergence than SGD for small datasets
- **Low Learning Rate (1e-4)**: Prevents overfitting and ensures stable training
- **Extended Patience (15)**: Allows model to escape local minima
- **Balanced Augmentation**: Increases dataset diversity without unrealistic distortions

## 📈 Training Process

### Complete Workflow Overview

#### Phase 1: Environment Setup (Google Colab)
1. **Create new Colab notebook** with GPU runtime
2. **Mount Google Drive** for persistent storage
3. **Upload dataset** (`Hackathon_Dataset.zip`) to Google Drive
4. **Install YOLOv8** via ultralytics package

#### Phase 2: Data Preparation
1. **Extract dataset** from zip file to temporary directory
2. **Verify data structure** and file integrity
3. **Generate YAML configuration** with correct paths
4. **Validate class distribution** across train/val/test splits

#### Phase 3: Training Execution
1. **Load pre-trained YOLOv8x** as starting point
2. **Configure hyperparameters** for optimal performance
3. **Execute training loop** with monitoring and validation
4. **Implement early stopping** to prevent overfitting

#### Phase 4: Model Evaluation
1. **Run comprehensive validation** on test set
2. **Generate performance metrics** (precision, recall, mAP)
3. **Create confusion matrix** for class-wise analysis
4. **Visualize predictions** on sample test images

#### Phase 5: Model Export
1. **Save best model weights** (`best.pt`)
2. **Download model** from Colab to local storage
3. **Document training results** and hyperparameters

### Key Training Features
- **Base Model**: YOLOv8x pre-trained weights (best accuracy variant)
- **Training Duration**: 100 epochs with early stopping (patience=15)
- **Image Resolution**: 896x896 pixels for high-detail detection
- **Batch Configuration**: 8 images per batch (T4 GPU optimized)
- **Optimization Strategy**: AdamW with learning rate 1e-4
- **Regularization**: Weight decay, dropout, label smoothing
- **Data Augmentation**: Mosaic, MixUp, HSV variations, geometric transforms

### Training Results Summary
- **Final Fitness Score**: 0.9549 (excellent overall performance)
- **Best Validation mAP50**: 97.70% (outstanding accuracy)
- **Training Convergence**: Achieved optimal performance around epoch 85
- **Hardware Utilization**: NVIDIA T4 GPU (Google Colab free tier)
- **Total Training Time**: Approximately 4-6 hours

### Google Drive Integration Benefits
- **Persistent Storage**: Dataset survives Colab session restarts
- **Easy Sharing**: Team members can access the same dataset
- **Version Control**: Keep multiple dataset versions organized
- **Backup Safety**: Automatic cloud backup of training data
- **Large File Support**: Handle datasets larger than Colab's local storage

## 🔍 Model Evaluation

### Confusion Matrix
The model shows excellent class separation with minimal confusion between classes:

- **FireExtinguisher**: High precision with some recall challenges
- **ToolBox**: Best performing class across all metrics
- **OxygenTank**: Balanced precision-recall performance

### Inference Speed
- **Preprocessing**: 1.4ms per image
- **Inference**: 73.6ms per image
- **Postprocessing**: 3.4ms per image
- **Total**: ~78.4ms per image

## 🎯 Use Cases

### Industrial Safety
- Automated safety equipment inspection
- Compliance monitoring in workplaces
- Emergency response planning

### Inventory Management
- Automated counting of safety equipment
- Location tracking and mapping
- Maintenance scheduling

### Quality Assurance
- Equipment condition assessment
- Placement verification
- Accessibility compliance

## 📋 Requirements

```txt
ultralytics>=8.0.0
opencv-python>=4.5.0
pillow>=8.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
pandas>=1.3.0
pyyaml>=5.4.0
torch>=1.9.0
torchvision>=0.10.0
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics**: For the amazing YOLOv8 implementation
- **Team Oink Boink**: For the collaborative development effort
- **HackByte Dataset**: For providing the training data
- **Google Colab**: For providing GPU resources for training

---

**Made with ❤️ by Team Oink Boink**
