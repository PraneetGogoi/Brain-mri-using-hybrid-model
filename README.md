
```markdown
# Brain MRI Using Hybrid Model

[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://github.com/PraneetGogoi/Brain-mri-using-hybrid-model)

This repository contains a **deep learning-based hybrid model** designed to classify brain MRI scans into different categories (e.g., tumor vs no tumor). It demonstrates data preprocessing, augmentation, model training, evaluation, and explainability visualization using state-of-the-art techniques. :contentReference[oaicite:1]{index=1}

---

## 🧠 Project Overview

Brain tumor classification using MRI is a crucial task in medical image analysis. Early and accurate automated classification can assist clinicians in making quicker and more reliable decisions.

This project leverages:

- Convolutional Neural Networks (CNNs)
- Transfer Learning (e.g., VGG, ResNet)
- Data Augmentation
- Model Evaluation Metrics (ROC, Confusion Matrix)
- Explainability Methods (Grad-CAM, Saliency Maps)

---

## 📁 Repository Structure

```

Brain-mri-using-hybrid-model/
├── augmented_images.png
├── confusion_matrix.png
├── gradcam_visualizations.png
├── gradcam_plus_plus_visualizations.png
├── model.ipynb
├── model1.ipynb
├── resnet50_training_history.png
├── roc_curves.png
├── saliency_map.png
├── training_history.png
└── metadata.csv

````

- **Notebooks:**  
  - `model.ipynb` — Core model training and evaluation  
  - `model1.ipynb` — Additional experiments and variations

- **Visual Outputs:**  
  - Augmented image samples  
  - Confusion matrices, ROC curves  
  - Visual explainability (Grad-CAM, saliency maps)

---

## 🚀 Getting Started

### 🛠️ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
````

If you don’t have a `requirements.txt`, install core libraries manually:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn shap
```

---

## 📌 Dataset Preparation

Before running the notebooks:

1. Place your MRI images in the appropriate folder structure:

   ```
   train/
   val/
   test/
   ```

2. Each folder should contain subfolders representing classes (e.g., `tumor`, `no_tumor`).

---

## 🧪 Training & Evaluation

* **Augmentation:** Images are augmented to improve generalization and prevent overfitting.
* **Model Architecture:** Uses CNN backbones with transfer learning (e.g., VGG16, ResNet50).
* **Metrics:** Accuracy, precision, recall, confusion matrix, ROC curves are plotted to analyze performance.

---

## 📊 Visualizations

This project includes saved visual explanations to understand model decisions:

| Visualization                | Description                            |
| ---------------------------- | -------------------------------------- |
| `augmented_images.png`       | Samples of augmented data              |
| `confusion_matrix.png`       | Performance summary                    |
| `roc_curves.png`             | ROC and AUC curves                     |
| `gradcam_visualizations.png` | Grad-CAM heatmaps                      |
| `saliency_map.png`           | Saliency maps showing pixel importance |

---

## 🧩 Explainability with SHAP

SHAP (SHapley Additive exPlanations) is used to interpret model predictions and visualize which regions of MRI images influence decisions most.

Install SHAP inside your Jupyter environment with:

```python
import sys
!{sys.executable} -m pip install shap
```

Then restart the kernel before import. ([GitHub][1])

---

## 🎯 Results

Trained models in this repo are evaluated using:

* **Confusion Matrix**
* **ROC AUC Scores**
* **Classification Reports**
* **Visual Explainability (Grad-CAM, SHAP)**

These help determine each class’s performance and interpret model decisions from an imaging perspective. ([GitHub][1])

---

## 📝 Contributing

Contributions are welcome! Feel free to:

* Add more models (EfficientNet, DenseNet, ViT)
* Improve augmentation strategies
* Add Docker support
* Add deployment pipeline (Flask, FastAPI, Streamlit)


## 💡 References

* Example MRI classification repository for reference: [Brain-MRI-Image-Classification-Using-Deep-Learning](https://github.com/strikersps/Brain-MRI-Image-Classification-Using-Deep-Learning) ([GitHub][1])

---

