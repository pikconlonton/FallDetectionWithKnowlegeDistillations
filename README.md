# FallDetectionWithKnowlegeDistillations

A comprehensive fall detection system that combines object detection and classification using knowledge distillation techniques to improve model performance,efficiency, model generalization.

## Overview

This project implements a two-stage fall detection system:
1. **Object Detection**: Uses YOLOv5 to detect human figures in images/videos
2. **Fall Classification**: Employs CNN models (ResNet-18, EfficientNet-B0) with knowledge distillation to classify detected regions as "fall" or "normal"

The knowledge distillation framework allows smaller student models to learn from larger teacher models (like Vision Transformer), achieving better performance with reduced computational requirements.
## Features

### 🎯 Two-Stage Detection Pipeline
- **Stage 1**: YOLOv5 detects human figures in input images/videos
- **Stage 2**: CNN classifiers determine if detected persons are falling

### 🧠 Knowledge Distillation
- **Teacher Model**: Vision Transformer (ViT-B/16) pre-trained and fine-tuned
- **Student Models**: ResNet-18 and EfficientNet-B0
- **Soft Label Generation**: Teacher model generates probability distributions
- **Distillation Loss**: Combines Cross-Entropy and KL-Divergence losses

### 📊 Multiple Model Architectures
- **ResNet-18**: Custom implementation with BasicBlock residual connections
- **EfficientNet-B0**: Custom implementation with MBConv blocks and SE attention
- **Vision Transformer**: Teacher model for knowledge distillation

### 🎥 Real-time Inference
- Live video processing capability
- Bounding box visualization
- Confidence score display

## Installation

1. Clone the repository:
```bash
git clone https://github.com/pikconlonton/FallDetectionWithKnowlegeDistillations.git
cd FallDetectionWithKnowlegeDistillations
```

2. Install dependencies:
```bash
pip install -r src/requirements.txt
```

3. Install YOLOv5:
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

## Usage

### Dataset Preparation

1. **Data Sources**: The project uses multiple fall detection datasets (see `doc/path_dataset.txt`)
2. **Data Processing**: Use `ProcessingDataAndFinetuningYolov5.ipynb` to:
   - Process YOLO format annotations
   - Extract bounding boxes for classification
   - Create train/validation splits

### Training

#### 1. Train YOLOv5 for Person Detection
```python
# Use ProcessingDataAndFinetuningYolov5.ipynb
# Fine-tune YOLOv5s on fall detection datasets
```

#### 2. Train Classification Models

**Without Knowledge Distillation:**
```python
# Use train_model.ipynb
# Train ResNet-18 or EfficientNet-B0 directly
```

**With Knowledge Distillation:**
```python
# Use train-with-kd.ipynb or make_soft_labels.py

# Step 1: Generate soft labels from teacher model
compute_and_save_soft_labels(teacher_model, train_dataset, save_path)

# Step 2: Train student model with soft labels
train_with_KD_soft_labels(student_model, train_dataset, test_dataloader, soft_labels_path)
```

### Inference

Run the complete pipeline:
```python
python src/Experiment.py
```

This will:
1. Load trained YOLOv5 and classification models
2. Process video input (camera or file)
3. Detect persons and classify fall/normal states
4. Display results with bounding boxes and labels

## Model Architectures

### ResNet-18
- Custom implementation with BasicBlock residual connections
- Batch normalization and ReLU activations
- Adaptive average pooling for flexible input sizes

### EfficientNet-B0
- MBConv blocks with depthwise separable convolutions
- Squeeze-and-Excitation (SE) attention modules
- Swish activation function
- Compound scaling methodology

### Knowledge Distillation Framework
- **Temperature Scaling**: T=2 for softening probability distributions
- **Loss Combination**: α=0.85 for hard labels, β=0.15 for soft labels
- **KL Divergence**: Measures difference between teacher and student outputs

## Dataset Information

The project uses multiple public fall detection datasets:
- Fall Detection datasets from Roboflow Universe
- UR Fall Detection Dataset
- HAR-UP Dataset
- Custom augmented datasets

**Dataset Structure:**
```
dataset_classification/
├── train/
│   ├── images/     # Cropped person images
│   └── labels.txt  # Binary labels (0: fall, 1: normal)
├── valid/
│   ├── images/
│   └── labels.txt
└── test/
    ├── images/
    └── labels.txt
```

## Results

The knowledge distillation approach typically achieves:
- **Improved Accuracy**: Student models perform better than baseline training
- **Model Compression**: Smaller models with competitive performance
- **Faster Inference**: Reduced computational requirements for deployment

## Technical Details

### Knowledge Distillation Process
1. **Teacher Training**: Fine-tune Vision Transformer on fall detection data
2. **Soft Label Generation**: Use teacher to generate probability distributions
3. **Student Training**: Train smaller models using both hard and soft labels
4. **Loss Function**: L = α·CE_loss + β·T²·KL_loss

### Data Augmentation
- Random affine transformations (rotation, translation, scaling)
- Color jittering (brightness, contrast, saturation, hue)
- ImageNet normalization for transfer learning compatibility

## Requirements

- Python 3.7+
- PyTorch 1.9+
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- tqdm
- Pillow

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is for educational and research purposes. Please cite relevant papers when using this code.

## References

- Vision Transformer (ViT) paper
- EfficientNet paper
- Knowledge Distillation papers
- YOLOv5 implementation
- ResNet architecture paper

See `doc/` folder for detailed research papers and references.

## Contact

For questions or collaboration, please open an issue in this repository.
