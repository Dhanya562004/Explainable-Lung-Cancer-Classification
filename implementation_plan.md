# Implementation Plan - Explainable Lung Cancer Classification

Refactor, enhance, train, evaluate, and deploy an explainable lung CT image classification model based on transfer learning with Xception, complete with a Grad-CAM explainability pipeline and a interactive Streamlit web interface.

## User Review Required

> [!IMPORTANT]
> - **Educational/Research Disclaimer**: The application explicitly features a disclamier stating it is not a medical diagnostic tool.
> - **Training Strategy**: A 2-stage transfer learning approach will be executed on the local machine using TensorFlow. Stage 1 trains the classification head with a frozen Xception backbone. Stage 2 unfreezes the top layers for low learning rate fine-tuning.
> - **Dataset Folder Standardization**: Subfolder names across train/valid/test will be standardized to `adenocarcinoma`, `large_cell_carcinoma`, `normal`, and `squamous_cell_carcinoma` for clean directory scanning and mapping.

## Proposed Changes

### Project Organization

Restructure repository into standard AI/ML layout:

```
project/
├── dataset/
│   ├── train/
│   ├── valid/
│   └── test/
├── notebooks/
│   └── model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── gradcam.py
├── models/
│   └── best_model.keras
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── evaluation_metrics.json
│   └── gradcam_sample.png
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

### Source Code (`src/`)

#### [NEW] [utils.py](file:///c:/Users/Deeksha/Downloads/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/src/utils.py)
- Configuration constants: `IMAGE_SIZE = (299, 299)`, `BATCH_SIZE = 16`, `NUM_CLASSES = 4`.
- Explicit class mapping dictionary:
  `{0: 'Adenocarcinoma', 1: 'Large Cell Carcinoma', 2: 'Normal', 3: 'Squamous Cell Carcinoma'}`.
- Data loading and generator builder helper functions (`get_data_generators`).

#### [NEW] [train.py](file:///c:/Users/Deeksha/Downloads/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/src/train.py)
- Implements Xception transfer learning model architecture:
  `Xception (pretrained ImageNet) -> GlobalAveragePooling2D -> BatchNormalization -> Dropout(0.4) -> Dense(4, softmax)`.
- Stage 1: Freeze base model, train head (`lr=1e-3`, Adam optimizer, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint).
- Stage 2: Unfreeze top Xception layers (block 13 & 14), fine-tune with low learning rate (`lr=1e-5`).
- Save final trained model to `models/best_model.keras`.
- Plot and save combined Stage 1 & Stage 2 loss and accuracy training curves to `results/training_curves.png`.

#### [NEW] [evaluate.py](file:///c:/Users/Deeksha/Downloads/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/src/evaluate.py)
- Evaluates model strictly on the `test` split.
- Computes Accuracy, Precision, Recall, Macro/Weighted F1-score, and Classification Report using `scikit-learn`.
- Plots and saves styled confusion matrix to `results/confusion_matrix.png`.
- Exports metrics to `results/evaluation_metrics.json`.

#### [NEW] [predict.py](file:///c:/Users/Deeksha/Downloads/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/src/predict.py)
- Loads `models/best_model.keras`.
- Preprocesses input image using Xception `preprocess_input`.
- Returns predicted class label, confidence percentage, and full probability breakdown for all 4 classes.

#### [NEW] [gradcam.py](file:///c:/Users/Deeksha/Downloads/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/src/gradcam.py)
- Implements Grad-CAM algorithm targeted at Xception's final convolutional layer (`block14_sepconv2_act`).
- Generates heatmap, overlays heatmap onto original CT scan using OpenCV jet colormap and alpha blending.
- Saves sample visualization to `results/gradcam_sample.png`.

---

### Web Application & Notebook

#### [NEW] [app.py](file:///c:/Users/Deeksha/Downloads/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/app.py)
- Interactive Streamlit dashboard.
- Features: Image file upload, prediction results display with confidence %, probability breakdown bar chart, and Grad-CAM visualization overlay.
- Prominent educational and non-clinical disclaimer.

#### [NEW] [notebooks/model_training.ipynb](file:///c:/Users/Deeksha/Downloads/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/notebooks/model_training.ipynb)
- Clean, reproducible Jupyter notebook containing full data loading, 2-stage training, evaluation, and Grad-CAM visualizations.

---

### Documentation & Repository Root

#### [MODIFY] [README.md](file:///c:/Users/Deeksha/Downloads/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/README.md)
- Complete rewrite for portfolio presentation with project overview, architecture diagrams, training curves, confusion matrix, Grad-CAM output, actual test metrics, local installation, and usage guide.

#### [NEW] [requirements.txt](file:///c:/Users/Deeksha/Downloads/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/requirements.txt)
- Exact clean dependency list.

## Verification Plan

### Automated Verification
- Run `python src/train.py` to train model and save weights & curves.
- Run `python src/evaluate.py` to calculate test metrics and confusion matrix.
- Run `python src/predict.py` with sample CT images to verify prediction and confidence outputs.
- Run `python src/gradcam.py` with sample CT image to verify heatmap generation.

### Manual Verification
- Launch Streamlit app using `streamlit run app.py` and test uploading sample images from the test dataset.
- Confirm Grad-CAM heatmap visualization and probability breakdown render cleanly on Streamlit UI.
