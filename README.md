# Explainable Lung Cancer Classification

## Overview
Explainable Lung Cancer Classification is an end-to-end computer vision and deep learning application designed to classify chest Computed Tomography (CT) scans into four distinct categories (Adenocarcinoma, Large Cell Carcinoma, Normal, and Squamous Cell Carcinoma) using Xception transfer learning, two-stage fine-tuning, and Grad-CAM visual explainability. I built the model training and explainability pipeline using TensorFlow/Keras, OpenCV, and Scikit-Learn, and deployed the interactive user interface using Streamlit.

With this application, users can upload a chest CT scan, get instant class predictions with model confidence percentages, view detailed probability breakdowns across all 4 categories, and inspect an interactive Grad-CAM heatmap overlay that visually highlights the exact anatomical regions driving the neural network's classification decision.

> ⚠️ **DISCLAIMER:** This project is strictly for **educational and research purposes only**. It is **not** a medical diagnostic tool and must **not** be used for clinical decision-making or patient diagnosis.

---

## Features
Here are the main features of the application:

- **Multi-Class CT Scan Classification**: Accurately classifies CT scans into 4 classes: Adenocarcinoma, Large Cell Carcinoma, Normal, and Squamous Cell Carcinoma.
- **Two-Stage Transfer Learning & Fine-Tuning**: Feature extraction stage with a frozen Xception backbone, followed by low learning-rate fine-tuning of top convolutional blocks (`block13` & `block14`).
- **Grad-CAM Visual Explainability**: Computes gradient-weighted class activation maps (`block14_sepconv2_act`) to generate jet-colormap heatmap overlays showing exact CT regions driving predictions.
- **Instant Confidence & Probability Breakdown**: Displays predicted class, percentage confidence, and individual bar charts for all 4 class probabilities.
- **Medically Appropriate Augmentation**: Applies small rotations, width/height shifts, zoom, and horizontal flips while preserving anatomical structure during training.
- **Interactive Streamlit Web Dashboard**: User-friendly web interface allowing instant CT upload, analysis execution, and side-by-side original/heatmap visual comparisons.
- **Comprehensive Evaluation Metrics**: Evaluated on an isolated test split ($N=315$) generating accuracy, precision, recall, F1-scores, and confusion matrix visualizations.

---

## Technologies Used
This project leverages a modern Python AI/ML and web stack to deliver an explainable image classification experience:

### Machine Learning & Deep Learning
- **TensorFlow (v2.21.0) & Keras** - Serves as the primary deep learning framework for building, training, fine-tuning, and saving the Xception model.
- **Xception Architecture** - Extreme Inception backbone pretrained on ImageNet used for high-level visual feature extraction.
- **Scikit-Learn** - Used for calculating classification reports, accuracy, precision, recall, macro/weighted F1-scores, and confusion matrices.

### Image & Data Processing
- **OpenCV (v5.0.0)** - Handles image array color conversions (RGB/BGR), heatmap resizing, Jet colormap application, and alpha blending.
- **Pillow (v12.3.0)** - Used for loading, resizing, and converting input image formats.
- **NumPy & Pandas** - Manages tensor manipulation, matrix operations, and metric aggregation.

### Visualization & Web Interface
- **Streamlit (v1.61.1)** - Powers the interactive web dashboard, handling file uploads, progress bars, metric cards, and image rendering tabs.
- **Matplotlib & Seaborn** - Generates and exports training loss/accuracy curves and confusion matrix heatmap plots.

---

## Project Structure
Here's how the repository is structured:

```
Explainable-Lung-Cancer-Classification/
├── dataset/                             # Chest CT-Scan Image Dataset
│   ├── train/                           # Training split (4 classes)
│   ├── valid/                           # Validation split (4 classes)
│   └── test/                            # Isolated test split (4 classes)
│
├── notebooks/                           # Interactive Notebooks
│   └── model_training.ipynb             # Full training & Grad-CAM walkthrough
│
├── src/                                 # Modular Source Code
│   ├── __init__.py                      # Package marker
│   ├── utils.py                         # Constants, class maps & data generators
│   ├── train.py                         # 2-Stage transfer learning pipeline
│   ├── evaluate.py                      # Test set evaluation & confusion matrix
│   ├── predict.py                       # Inference engine & class probabilities
│   └── gradcam.py                       # Grad-CAM heatmap overlay generator
│
├── models/                              # Trained Model Artifacts
│   └── best_model.keras                 # Saved model weights (excluded from git)
│
├── results/                             # Generated Performance Artifacts
│   ├── training_curves.png              # Loss & accuracy epoch curves
│   ├── confusion_matrix.png             # Test set confusion matrix heatmap
│   ├── evaluation_metrics.json          # Numerical metric breakdown
│   └── gradcam_sample.png               # Sample Grad-CAM overlay image
│
├── app.py                               # Interactive Streamlit Web Application
├── requirements.txt                     # Python dependencies
├── README.md                            # Main project documentation
└── LICENSE                              # MIT License
```

---

## Installation & Setup

### Prerequisites
Before getting started, ensure you have the following installed on your machine:
- **Python (v3.10 or newer)**
- **Git**

### Step 1: Clone the Repository
```bash
git clone https://github.com/Dhanya562004/Explainable-Lung-Cancer-Classification.git
cd Explainable-Lung-Cancer-Classification
```

### Step 2: Set up Virtual Environment & Install Dependencies
Create a virtual environment and install the required packages:

```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Activate environment (Linux/macOS)
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Step 3: Train the Model (Optional)
To re-run the two-stage transfer learning and fine-tuning pipeline:
```bash
python src/train.py
```

### Step 4: Evaluate the Model
To generate test set metrics and confusion matrix plots:
```bash
python src/evaluate.py
```

### Step 5: Start the Streamlit Web App
To launch the interactive user interface:
```bash
streamlit run app.py
```
This will open the application in your default browser at `http://localhost:8501`.

---

## Usage

- **As a Student / Researcher**: Explore `notebooks/model_training.ipynb` or run scripts under `src/` to inspect data generators, two-stage fine-tuning callbacks, and Grad-CAM tensor gradient computations.
- **As a Web Dashboard User**: Open the Streamlit app, upload a chest CT scan (PNG/JPG), click **Analyze Image**, and view the predicted cancer class, confidence percentage, probability breakdown, and interactive Grad-CAM visual overlay.

---

## Grad-CAM Visual Explainability

To provide transparency into model predictions, the application implements **Gradient-weighted Class Activation Mapping (Grad-CAM)**:

1. Extracts feature map activations from Xception's final convolutional layer (`block14_sepconv2_act`).
2. Calculates gradients of the predicted class score with respect to target feature maps using `tf.GradientTape()`.
3. Pools gradient intensity values to weight feature map channels.
4. Applies Rectified Linear Unit (ReLU) filtering to highlight positive contributions.
5. Superimposes an OpenCV `COLORMAP_JET` heatmap over the original CT scan ($40\%$ heatmap opacity $+ 60\%$ original scan).

Warm regions (red/yellow) indicate key visual feature areas in the CT scan driving the neural network's classification decision.

---

## Model Evaluation & Performance

Evaluated on the **isolated test set** ($N = 315$ CT images):

| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | **76.51%** |
| **Macro Precision** | **0.8063** |
| **Macro Recall** | **0.7983** |
| **Macro F1-Score** | **0.7859** |
| **Weighted F1-Score** | **0.7623** |

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Test Samples |
| :--- | :---: | :---: | :---: | :---: |
| **Adenocarcinoma** | 0.90 | 0.57 | 0.70 | 120 |
| **Large Cell Carcinoma** | 0.77 | 0.73 | 0.75 | 51 |
| **Normal** | **0.95** | **0.98** | **0.96** | 54 |
| **Squamous Cell Carcinoma** | 0.61 | 0.91 | 0.73 | 90 |

---

## Future Improvements
There are a few key features to add in future iterations:

- **3D CT Volumetric Support**: Extending 2D slice processing to handle 3D NIfTI (`.nii`) and DICOM (`.dcm`) series files.
- **Ensemble Architecutres**: Combining predictions from Xception, EfficientNetV2, and ResNet50V2 backbones for higher accuracy.
- **Cloud API Deployment**: Containerizing the model with Docker and serving REST API endpoints via FastAPI.
- **Interactive Heatmap Threshold Slider**: Adding a Streamlit control slider to dynamically adjust Grad-CAM activation thresholds.

---

## Challenges & Learning

- **Fine-Tuning Deep Convolutional Backbones**: Balancing frozen feature extraction (Stage 1) and low learning-rate unfreezing (Stage 2) without causing catastrophic forgetting.
- **Grad-CAM Activation Mapping in Keras 3**: Constructing sub-models with gradient tapes to extract convolutional activations cleanly without modifying original model graphs.
- **Medical Data Augmentation Constraints**: Designing transformations (rotation, shift, zoom, horizontal flip) that augment training variance without distorting chest anatomical geometry.

---

## Contributing
Contributions are welcome! If you have suggestions or find bugs, feel free to contribute:

1. Fork this repository.
2. Create a new branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add new feature'`
4. Push to your branch: `git push origin feature-name`
5. Open a Pull Request.

---

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details. Original work acknowledged by Rhitam Chaudhury.

---

## Author
**Dhanya K**  
*AI/ML Student & Developer*  
GitHub: [@Dhanya562004](https://github.com/Dhanya562004)
