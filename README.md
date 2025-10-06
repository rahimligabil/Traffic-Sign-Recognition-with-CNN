# Traffic-Sign-Recognition-with-CNN
Traffic sign recognition using a custom CNN trained on GTSRB dataset (PyTorch implementation with metrics, evaluation and confusion matrix).



This project implements a **Convolutional Neural Network (CNN)** model for classifying traffic signs from the **GTSRB dataset**.  
The model is built with PyTorch and includes data preprocessing, training, and evaluation phases.

---

## Overview
Traffic sign recognition is an essential part of autonomous driving and driver-assistance systems.  
In this project, a compact CNN model is trained on 43 traffic sign classes using the GTSRB dataset.  
The model takes 30×30 RGB images as input and predicts the sign category.

**Accuracy:** 99.7%

---

## 🧱 Model Architecture
The CNN model includes:
- Convolution + ReLU + MaxPooling layers
- Fully connected (Linear) layers
- Softmax activation at output
