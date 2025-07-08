# Pneumonia Detection using InceptionV3 on the MedMNIST PneumoniaMNIST Dataset

This project builds a deep learning model using **InceptionV3** to detect **pneumonia** from chest X-ray images using the **PneumoniaMNIST** dataset from [MedMNIST](https://medmnist.com/). The model is optimized for performance and memory efficiency, and includes data augmentation, class balancing, and multiple evaluation metrics.

---

## 📌 Table of Contents

- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Evaluation](#evaluation)
- [Performance Metrics](#performance-metrics)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Results](#results)
- [License](#license)

---

## 🧠 Dataset

We use the [PneumoniaMNIST](https://medmnist.com/) dataset (part of MedMNIST v2), which contains grayscale chest X-ray images categorized into two classes:

- **0**: Normal  
- **1**: Pneumonia

### Dataset Split:

- **Training set**: 4708 images  
- **Validation set**: 936 images  
- **Test set**: 1560 images

Each image is resized to **128×128** and converted to **RGB** before training.

---

## 🏗️ Model Architecture

We use **InceptionV3** from Keras Applications with pretrained `ImageNet` weights (without top layers), and add a custom classification head:

- Input size: `128x128x3`
- **Base**: InceptionV3 (frozen during training)
- **Head**:
  - `GlobalAveragePooling2D`
  - `Dropout(0.7)`
  - `Dense(64, relu)` with L2 regularization
  - `Dense(2, softmax)` for binary classification

---

## ⚙️ Training Details

- **Loss**: Categorical Crossentropy
- **Optimizer**: Adam (`lr=1e-4`)
- **Batch size**: 4 (RAM-optimized)
- **Epochs**: 20 (with early stopping)
- **Callbacks**:
  - `EarlyStopping`
  - `ReduceLROnPlateau`
  - `MemoryCleanup` (custom callback for garbage collection)
- **Data Augmentation**:
  - Random horizontal flip
  - Rotation range: 10 degrees
  - Zoom range: 0.1

- **Class weights** are computed to handle class imbalance.

---

## 📈 Evaluation

After training, the model is evaluated on the test set using the following metrics:

- Accuracy
- AUC (Area Under Curve)
- Precision
- Recall
- Confusion Matrix
- ROC Curve
- Classification Report

---

## 📊 Performance Metrics

| Metric     | Value     |
|------------|-----------|
| Accuracy   | ~XX.XX%   |
| AUC        | ~X.XXXX   |
| Precision  | ~XX.XX%   |
| Recall     | ~XX.XX%   |

> **Note:** Exact numbers depend on your hardware, training epochs, and data randomization.

---

## ▶️ How to Run

### 1. Clone the repository:

```bash
git clone https://github.com/Harveer-Kaur/pneumonia-inceptionv3.git
cd pneumonia-inceptionv3
### 2. Install dependency 
pip3 install -r requirements.txt
### 3. Download pneumoniamnist.npz from MedMNIST PneumoniaMNIST and place it in the project root.
python3 train-pneumonia-model.py


