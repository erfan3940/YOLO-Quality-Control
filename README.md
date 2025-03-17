# YOLO-Quality-Control
# Object Detection and Quality Control with YOLOv11

![Project Banner](https://github.com/erfan3940/YOLO-Quality-Control/blob/main/screenshots/screen.png) <!-- Replace with your image/video thumbnail -->

This project focuses on implementing **YOLOv11** (You Only Look Once version 11) for real-time object detection in manufacturing lines. The goal is to identify objects, classify them as either **damaged** or **OK**, and count them efficiently. This system can be integrated into production lines to automate quality control processes.

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Usage](#usage)
6. [Demo](#demo)
7. [Acknowledgments](#acknowledgments)

---

## Overview

The manufacturing industry requires efficient quality control systems to ensure product consistency and reduce waste. In this project, we leverage **YOLOv11**, a state-of-the-art object detection model, to:

- Detect objects on the production line.
- Classify them as either **damaged** or **OK**.
- Count the number of objects in each category.

This solution is designed to be fast, accurate, and scalable for industrial applications.

---

## Dataset

The dataset used in this project was sourced from **Kaggle**. It contains labeled images of objects captured in various conditions on a manufacturing line. Each object is annotated with bounding boxes and classified into one of two categories:
- **Damaged**: Objects with visible defects.
- **OK**: Objects that meet quality standards.

Key Details:
- **Dataset Name**: [Insert Kaggle Dataset Name Here](link_to_kaggle_dataset)
- **Number of Images**: X,XXX
- **Annotations**: Bounding boxes and labels (Damaged/OK)
- **Data Augmentation**: Applied to improve model robustness (e.g., rotation, scaling, flipping).

---

## Methodology

### Model Architecture
We used **YOLOv11**, an advanced real-time object detection model known for its speed and accuracy. Key features include:
- Single-shot detection for high-speed inference.
- Support for multiple classes (in this case, "Damaged" and "OK").
- Efficient handling of small and large objects.

### Training Process
1. **Preprocessing**:
   - Resized images to a fixed resolution (e.g., 640x640).
   - Normalized pixel values to [0, 1].
   - Split data into training (80%), validation (10%), and testing (10%) sets.
2. **Training**:
   - Fine-tuned YOLOv11 on the custom dataset.
   - Used transfer learning to leverage pre-trained weights.
   - Optimized with **AdamW** optimizer and a learning rate scheduler.
3. **Evaluation**:
   - Metrics: Precision, Recall, mAP (mean Average Precision), and F1-score.
   - Achieved **XX% mAP** on the test set.

### Deployment
The trained model was deployed on a production line simulator using Python and OpenCV for real-time inference.

---

## Results

### Performance Metrics
| Metric         | Value   |
|----------------|---------|
| Precision      | XX.XX%  |
| Recall         | XX.XX%  |
| mAP@0.5        | XX.XX%  |
| F1-Score       | XX.XX%  |

### Example Output
Below is an example of the model's output on a test image:

![Example Output](path_to_example_output_image.jpg) <!-- Replace with your actual image -->

---

## Usage

### Prerequisites
- Python 3.x
- Install dependencies:
  ```bash
  pip install torch torchvision opencv-python ultralytics
