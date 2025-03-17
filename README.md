# YOLO-Quality-Control
# Object Detection and Quality Control with YOLOv11

![Project Banner](https://github.com/erfan3940/YOLO-Quality-Control/blob/main/screenshots/screen.png)

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

The dataset used in this project was sourced from **Kaggle**. It contains images of objects captured in various conditions. Each object is annotated with bounding boxes and classified into one of two categories:
- **Damaged**: Objects with visible defects.
- **OK**: Objects that meet quality standards.

Key Details:
- **Dataset Name**: [https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product](link_to_kaggle_dataset)
- **Number of Images**: 700
- **Annotations**: Bounding boxes and labels (Damaged/OK) or in persian (سالم/خراب)

---

## Methodology

### Model Architecture
We used **YOLO11l and YOLO11s**, an advanced real-time object detection model known for its speed and accuracy. Key features include:
- Single-shot detection for high-speed inference.
- Support for multiple classes (in this case, "Damaged" and "OK").

### Training Process
1. **Preprocessing**:
   - Resized images to a fixed resolution (e.g., 512x512).
   - Split data into training (80%), validation (10%), and testing (10%) sets.
2. **Training**:
   - Fine-tuned YOLO11 on the custom dataset.
   - Used transfer learning to leverage pre-trained weights.
3. **Evaluation**:
   - Metrics: Precision, Recall, mAP (mean Average Precision), and F1-score.
   - Achieved **0.95% mAP** on the test set.

### Deployment
The trained model was deployed on a production line simulator using Python and OpenCV for real-time inference.

---

## Results

### Performance Metrics
| Metric         | Value   |
|----------------|---------|
| Precision      | 95.55%  |
| Recall         | 95.65%  |
| mAP@0.5        | 95.10%  |
| F1-Score       | 95.30%  |

### Example Output
Below is an short video example of the model's output on a test images:

![Example Output](https://github.com/erfan3940/YOLO-Quality-Control/blob/main/VideoRecorder/recorded1.avi)

---

## Usage

### Prerequisites
- Python 3.10.16
- Install dependencies:
  ```bash
  pip install torch torchvision opencv-python ultralytics
