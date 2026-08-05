import os
import sys
from PIL import Image
import numpy as np
import streamlit as st

# Add src to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import DEFAULT_MODEL_PATH, CLASS_LABELS
from predict import get_model
from gradcam import generate_gradcam_overlay

st.set_page_config(
    page_title="Explainable Lung Cancer Classification",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 12px 16px;
        border-radius: 6px;
        color: #991B1B;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .pred-class-text {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
    }
    .confidence-text {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2563EB;
    }
    </style>
""", unsafe_allow_globals=True)

# Main Title Header
st.markdown('<div class="main-title">🫁 Explainable Lung Cancer Classification</div>', unsafe_allow_globals=True)
st.markdown('<div class="sub-title">Deep Transfer Learning with Xception Backbone & Grad-CAM Visual Explainability</div>', unsafe_allow_globals=True)

# Medical Disclaimer Banner
st.markdown("""
<div class="disclaimer-box">
    ⚠️ <strong>DISCLAIMER:</strong> For educational and research purposes only. 
    This application is not a medical diagnostic tool and should not be used for clinical decision-making or patient diagnosis.
</div>
""", unsafe_allow_globals=True)

# Sidebar
with st.sidebar:
    st.header("📌 Project Details")
    st.markdown("""
    **Model Backbone:** Xception (Pretrained on ImageNet)  
    **Fine-Tuning:** 2-Stage Transfer Learning  
    **Explainability:** Grad-CAM Heatmap  
    **Target Classes:**
    - Adenocarcinoma
    - Large Cell Carcinoma
    - Normal
    - Squamous Cell Carcinoma
    """)
    st.divider()
    st.markdown("### ⚙️ System Status")
    if os.path.exists(DEFAULT_MODEL_PATH):
        st.success("Trained model weights loaded (`best_model.keras`)")
    else:
        st.warning("⚠️ Trained model file not found (`models/best_model.keras`). Please run training first.")

# Main Interface
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Upload Chest CT Scan")
    uploaded_file = st.file_uploader(
        "Select a lung CT image (PNG, JPG, JPEG):",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded CT Image", use_container_width=True)
        analyze_btn = st.button("🔍 Analyze Image & Generate Explainability Map", use_container_width=True, type="primary")
    else:
        st.info("Please upload a CT scan image to begin analysis.")
        analyze_btn = False

with col_right:
    st.subheader("2. Prediction & Model Insight")

    if uploaded_file is not None and analyze_btn:
        if not os.path.exists(DEFAULT_MODEL_PATH):
            st.error("Model weights file not found. Please train the model using `python src/train.py` before running predictions.")
        else:
            with st.spinner("Executing model inference and generating Grad-CAM explainability heatmap..."):
                try:
                    model = get_model(DEFAULT_MODEL_PATH)
                    res = generate_gradcam_overlay(image, model=model)

                    # Prediction Summary Cards
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown('<div class="metric-card">', unsafe_allow_globals=True)
                        st.caption("Predicted Class")
                        st.markdown(f'<div class="pred-class-text">{res["predicted_class"]}</div>', unsafe_allow_globals=True)
                        st.markdown('</div>', unsafe_allow_globals=True)

                    with c2:
                        st.markdown('<div class="metric-card">', unsafe_allow_globals=True)
                        st.caption("Model Confidence")
                        st.markdown(f'<div class="confidence-text">{res["confidence"]:.2f}%</div>', unsafe_allow_globals=True)
                        st.markdown('</div>', unsafe_allow_globals=True)

                    st.markdown("### 📊 Class Probability Breakdown")
                    for cls_name, prob in res['probabilities'].items():
                        st.write(f"**{cls_name}**: `{prob:.2f}%`")
                        st.progress(float(prob / 100.0))

                    st.markdown("### 🔬 Grad-CAM Heatability Visualizer")
                    st.caption("Warm regions (red/yellow) indicate the key image features that influenced the model's prediction.")

                    tab1, tab2, tab3 = st.tabs(["Superimposed Overlay", "Heatmap Only", "Original Scan"])
                    with tab1:
                        st.image(res['superimposed_image'], caption="Grad-CAM Overlay on CT Image", use_container_width=True)
                    with tab2:
                        st.image(res['heatmap_colored'], caption="Grad-CAM Activation Heatmap", use_container_width=True)
                    with tab3:
                        st.image(res['original_image'], caption="Original Uploaded CT Image", use_container_width=True)

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.exception(e)
    elif uploaded_file is not None:
        st.info("Click 'Analyze Image & Generate Explainability Map' to see predictions.")

st.divider()
st.caption("Explainable Lung Cancer Classification Portfolio Project | Educational & Research Purpose Only")
