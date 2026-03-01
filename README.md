# 🫁 Pneumonia Detection from Chest X-Rays

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow"/>
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  A deep learning project for automated pneumonia detection from chest X-ray images, implementing and comparing two powerful CNN architectures — <strong>VGG16</strong> and <strong>DenseNet121</strong> — using transfer learning.
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Models](#-models)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Technologies Used](#-technologies-used)
- [License](#-license)

---

## 🔍 Overview

Pneumonia is a life-threatening lung infection that affects millions of people worldwide. Early and accurate detection is critical for effective treatment. This project leverages the power of **Convolutional Neural Networks (CNNs)** and **transfer learning** to classify chest X-ray images as either **Normal** or **Pneumonia**.

Two pre-trained models are implemented and compared:
- **VGG16** — A classic deep CNN known for its simplicity and strong feature extraction
- **DenseNet121** — A densely connected network that reuses features across layers for improved performance

---

## 📂 Dataset

The dataset used is the **Chest X-Ray Images (Pneumonia)** dataset, which contains labeled X-ray images organized into training, validation, and test sets.

| Split      | Normal | Pneumonia |
|------------|--------|-----------|
| Train      | 1,341  | 3,875     |
| Validation | 8      | 8         |
| Test       | 234    | 390       |

> 📌 Dataset source: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

---

## 🤖 Models

### 1. VGG16
VGG16 is a 16-layer deep convolutional neural network originally trained on ImageNet. In this project, the convolutional base is frozen and a custom classification head is added for binary classification.

- **Input Size:** 224 × 224 × 3
- **Pre-trained on:** ImageNet
- **Fine-tuned for:** Binary Classification (Normal vs Pneumonia)

### 2. DenseNet121
DenseNet121 uses dense connections between layers, where each layer receives feature maps from all preceding layers. This reduces vanishing gradients and improves feature propagation.

- **Input Size:** 224 × 224 × 3
- **Pre-trained on:** ImageNet
- **Fine-tuned for:** Binary Classification (Normal vs Pneumonia)

### Model Architecture Summary

| Feature              | VGG16         | DenseNet121      |
|----------------------|---------------|------------------|
| Depth                | 16 layers     | 121 layers       |
| Parameters           | ~138M         | ~8M              |
| Dense Connections    | ❌            | ✅               |
| Transfer Learning    | ✅            | ✅               |
| Data Augmentation    | ✅            | ✅               |

---

## 📁 Project Structure

```
xray/
│
├── data/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
├── models/
│   ├── vgg16_model.h5
│   └── densenet121_model.h5
│
├── notebooks/
│   ├── VGG16_Pneumonia_Detection.ipynb
│   └── DenseNet121_Pneumonia_Detection.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_vgg16.py
│   ├── train_densenet121.py
│   └── evaluate.py
│
├── results/
│   ├── vgg16_results/
│   └── densenet121_results/
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/xray.git
cd xray
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download the dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and place it in the `data/` directory.

---

## 🚀 Usage

### Train VGG16 Model
```bash
python src/train_vgg16.py
```

### Train DenseNet121 Model
```bash
python src/train_densenet121.py
```

### Evaluate Models
```bash
python src/evaluate.py --model vgg16
python src/evaluate.py --model densenet121
```

### Run Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

---

## 📊 Results

| Metric       | VGG16   | DenseNet121 |
|--------------|---------|-------------|
| Accuracy     | ~90%    | ~93%        |
| Precision    | ~89%    | ~92%        |
| Recall       | ~92%    | ~95%        |
| F1-Score     | ~90%    | ~93%        |
| AUC-ROC      | ~95%    | ~97%        |

> ⚠️ Results may vary depending on hyperparameters, training epochs, and dataset split.

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **TensorFlow / Keras** — Model building and training
- **NumPy / Pandas** — Data manipulation
- **Matplotlib / Seaborn** — Visualization
- **Scikit-learn** — Evaluation metrics
- **OpenCV** — Image preprocessing
- **Jupyter Notebook** — Experimentation

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

- [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [VGG16 Paper — Simonyan & Zisserman, 2014](https://arxiv.org/abs/1409.1556)
- [DenseNet Paper — Huang et al., 2016](https://arxiv.org/abs/1608.06993)

---

<p align="center">Made with ❤️ for medical AI research</p>
