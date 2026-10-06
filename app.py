import os
import sys
import json
import time
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import (
    MEDICAL_DISCLAIMER, CLASS_LABELS, CLASS_NAMES, DEFAULT_MODEL_PATH,
    DEFAULT_RESULTS_DIR, DEFAULT_DATASET_DIR
)
from src.models.inference import get_model, predict_ct_scan
from src.explainability.gradcam import generate_gradcam_overlay
from src.models.model_registry import get_active_model_info, load_model_registry
from src.evaluation.evaluate import evaluate_model
from src.data.ingestion import scan_dataset
from src.data.validation import validate_dataset
from src.features.feature_pipeline import FeatureExtractor
from src.monitoring.latency import get_latency_stats
from src.monitoring.drift import compute_image_stats, detect_data_drift
from src.monitoring.performance import log_ground_truth_feedback, get_verified_performance_metrics
from src.monitoring.health import check_system_health
from src.logging.audit_logger import get_recent_audit_logs

# Streamlit Page Setup
st.set_page_config(
    page_title="Explainable Lung Cancer Classification",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 12px 16px;
        border-radius: 6px;
        color: #991B1B;
        font-size: 0.88rem;
        font-weight: 500;
        margin-bottom: 20px;
    }
    .card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 15px;
    }
    .badge-high {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-med {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title Header
st.markdown('<div class="main-title">🫁 Explainable Lung Cancer Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Production-Style MLOps Infrastructure | Deep Transfer Learning (Xception) & Grad-CAM Explainability</div>', unsafe_allow_html=True)

# Medical Disclaimer Banner
st.markdown(f'<div class="disclaimer-box">{MEDICAL_DISCLAIMER}</div>', unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/lungs.png", width=60)
    st.title("Navigation")
    nav_option = st.radio(
        "Select System Module:",
        [
            "🏠 Home",
            "🫁 CT Scan Prediction",
            "🔬 Grad-CAM Explainability",
            "⚙️ Model Information",
            "📊 Model Evaluation",
            "🏥 Dataset Health",
            "🧬 Feature / Representation Analysis",
            "📈 Monitoring Dashboard",
            "📉 Drift Monitoring",
            "📝 Performance Feedback",
            "🛡️ Responsible AI",
            "🩺 System Health"
        ]
    )
    
    st.divider()
    active_model = get_active_model_info()
    st.caption(f"**Active Model:** `{active_model.get('version', 'v1.0.0-xception')}`")
    st.caption(f"**Backbone:** Xception (ImageNet)")
    st.caption(f"**Test Accuracy:** `{active_model.get('metrics', {}).get('test_accuracy', 0.7651)*100:.2f}%`")

# Shared Cached Model Access
@st.cache_resource(show_spinner=False)
def load_shared_model():
    return get_model()

# ==========================================
# 1. 🏠 HOME
# ==========================================
if nav_option == "🏠 Home":
    st.header("🏠 Project Overview & Production Architecture")
    
    st.markdown("""
    This project is an **end-to-end Machine Learning Engineering system** specifically aligned with the **UnitedHealth Group / Optum AI/ML Engineer** job requirements. 
    It upgrades an existing 4-class Xception CT lung cancer classifier into a production-oriented MLOps platform featuring dataset validation, REST API capabilities, drift monitoring, confidence thresholds, and Grad-CAM interpretability.
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Model Backbone</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">Xception</div>', unsafe_allow_html=True)
        st.caption("Deep Transfer Learning")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Test Accuracy</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">76.51%</div>', unsafe_allow_html=True)
        st.caption("4-Class Evaluation")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Macro F1 Score</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">0.7859</div>', unsafe_allow_html=True)
        st.caption("Balanced Evaluation")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Explainability</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">Grad-CAM</div>', unsafe_allow_html=True)
        st.caption("Visual Interpretability")
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("🎯 Optum AI/ML Job Alignment Highlights")
    st.markdown("""
    - **ML Development:** 4-class CT classification (Adenocarcinoma, Large Cell Carcinoma, Normal, Squamous Cell Carcinoma) using Xception fine-tuning.
    - **ML Lifecycle & MLOps:** Reusable data pipelines, FastAPI REST endpoints (`/predict`, `/health`, `/metrics`), model registry versioning, and retraining promotion gates.
    - **Model Monitoring:** Real-time latency tracking, prediction confidence categorization (HIGH / MEDIUM / LOW), input distribution statistical drift detection, and verified ground-truth performance logging.
    - **Responsible AI & Healthcare:** Grad-CAM visual heatmaps, human oversight triggers for low-confidence inferences, privacy compliance (no raw patient image storage), and non-clinical disclaimers.
    """)

# ==========================================
# 2. 🫁 CT SCAN PREDICTION
# ==========================================
elif nav_option == "🫁 CT Scan Prediction":
    st.header("🫁 CT Scan Prediction Service")
    st.markdown("Upload a chest CT image for multi-class classification and confidence scoring.")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        uploaded_file = st.file_uploader("Select a Chest CT Scan Image (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded CT Image", use_container_width=True)
            analyze_btn = st.button("🔍 Run Prediction", type="primary", use_container_width=True)
        else:
            st.info("Upload a CT scan image above to begin.")
            analyze_btn = False

    with col_right:
        if uploaded_file is not None and analyze_btn:
            with st.spinner("Executing model inference pipeline..."):
                model = load_shared_model()
                res = predict_ct_scan(image, model=model)

                st.subheader("Results Summary")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.caption("Predicted Class")
                    st.markdown(f"### **{res['predicted_class']}**")
                    st.markdown('</div>', unsafe_allow_html=True)

                with c2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.caption("Confidence Score")
                    badge_cls = "badge-high" if res['confidence_level'] == "HIGH" else ("badge-med" if res['confidence_level'] == "MEDIUM" else "badge-low")
                    st.markdown(f"### **{res['confidence']:.2f}%** <span class='{badge_cls}'>{res['confidence_level']}</span>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                if res['requires_human_review']:
                    st.warning(f"⚠️ **HUMAN REVIEW RECOMMENDED:** {res['recommendation']}")

                st.markdown(f"⏱️ **Inference Latency:** `{res['inference_latency_ms']:.1f} ms` | 🏷️ **Model Version:** `{res['model_version']}` | 🆔 `{res['prediction_id']}`")

                st.subheader("📊 Class Probability Breakdown")
                for cls_name, prob in res['probabilities'].items():
                    st.write(f"**{cls_name}**: `{prob:.2f}%`")
                    st.progress(float(prob / 100.0))

# ==========================================
# 3. 🔬 GRAD-CAM EXPLAINABILITY
# ==========================================
elif nav_option == "🔬 Grad-CAM Explainability":
    st.header("🔬 Visual Interpretability via Grad-CAM")
    st.markdown("Grad-CAM highlights the key anatomical regions of interest in the chest CT scan that contributed to the model's prediction.")

    uploaded_file = st.file_uploader("Upload CT Image for Visual Explanation:", type=["png", "jpg", "jpeg"], key="gradcam_upload")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        if st.button("Generate Heatmap Overlay", type="primary"):
            with st.spinner("Computing gradient activations from Xception convolutional feature map..."):
                model = load_shared_model()
                res = generate_gradcam_overlay(image, model=model)

                st.success(f"Grad-CAM Generated for **{res['predicted_class']}** ({res['confidence']:.1f}% confidence)")

                t1, t2, t3 = st.tabs(["Superimposed Overlay", "Activation Heatmap", "Original Scan"])
                with t1:
                    st.image(res['superimposed_image'], caption="Grad-CAM Jet Overlay on CT Image", use_container_width=True)
                with t2:
                    st.image(res['heatmap_colored'], caption="Activation Heatmap Only", use_container_width=True)
                with t3:
                    st.image(res['original_image'], caption="Original CT Scan", use_container_width=True)
    else:
        st.info("Upload an image above to generate Grad-CAM visual explanations.")

# ==========================================
# 4. ⚙️ MODEL INFORMATION
# ==========================================
elif nav_option == "⚙️ Model Information":
    st.header("⚙️ Model Architecture & Version Registry")
    info = get_active_model_info()

    st.subheader(f"Active Version: `{info.get('version', 'v1.0.0-xception')}`")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Architecture Specs:**")
        st.write(f"- **Backbone:** {info.get('architecture', 'Xception Transfer Learning')}")
        st.write(f"- **Base Weights:** ImageNet Pre-trained")
        st.write(f"- **Total Parameters:** `{info.get('num_parameters', 21092804):,}`")
        st.write(f"- **Trainable Parameters:** `{info.get('trainable_parameters', 3613060):,}`")
        st.write(f"- **Input Dimensions:** `299 x 299 x 3` RGB")
    with col2:
        st.markdown("**Training Hyperparameters:**")
        cfg = info.get('training_config', {})
        st.json(cfg)

    st.subheader("Model Registry History")
    reg_all = load_model_registry()
    st.json(reg_all)

# ==========================================
# 5. 📊 MODEL EVALUATION
# ==========================================
elif nav_option == "📊 Model Evaluation":
    st.header("📊 Model Evaluation & Metrics")
    st.markdown("Actual measured performance metrics computed on the isolated 315-image test split.")

    metrics = evaluate_model()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Accuracy", f"{metrics['test_accuracy']*100:.2f}%")
    c2.metric("Macro Precision", f"{metrics['macro_precision']:.4f}")
    c3.metric("Macro Recall", f"{metrics['macro_recall']:.4f}")
    c4.metric("Macro F1-Score", f"{metrics['macro_f1_score']:.4f}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Confusion Matrix")
        cm_path = os.path.join(DEFAULT_RESULTS_DIR, 'confusion_matrix.png')
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.write(metrics.get('confusion_matrix'))

    with col_right:
        st.subheader("Per-Class Metrics Report")
        rep = metrics.get('classification_report', {})
        st.json(rep)

# ==========================================
# 6. 🏥 DATASET HEALTH
# ==========================================
elif nav_option == "🏥 Dataset Health":
    st.header("🏥 Dataset Ingestion & Health Summary")
    st.markdown("Automated scanner inspecting train/validation/test directory splits and checking file integrity.")

    if st.button("Run Dataset Scan & Health Validation", type="primary"):
        with st.spinner("Scanning dataset directory..."):
            report = validate_dataset()
            st.subheader(f"Validation Status: `{report['status']}`")

            if report['errors']:
                for err in report['errors']:
                    st.error(err)
            if report['warnings']:
                for wrn in report['warnings']:
                    st.warning(wrn)

            summary = report['summary']
            st.write(f"**Total Valid Images:** `{summary['total_images']}` | **Corrupt Files:** `{summary['corrupt_images']}`")
            st.write(f"**Image Dimensions Sample:** `{', '.join(summary['dimensions_sample'])}`")

            st.json(summary['splits'])

# ==========================================
# 7. 🧬 FEATURE / REPRESENTATION ANALYSIS
# ==========================================
elif nav_option == "🧬 Feature / Representation Analysis":
    st.header("🧬 Feature Representation Analysis")
    st.markdown("Inspect 2048-dimensional intermediate feature embeddings extracted from Xception backbone.")

    if st.button("Analyze Sample Feature Embeddings", type="primary"):
        with st.spinner("Extracting 2048-dim feature vectors from sample images..."):
            extractor = FeatureExtractor()
            res = extractor.analyze_dataset_embeddings(DEFAULT_DATASET_DIR, max_samples_per_class=5)

            st.write(f"**Embedding Dimensionality:** `{res['embedding_dim']}`")
            st.write(f"**Total Analyzed Images:** `{res['total_analyzed']}`")
            st.write(f"**Overall Feature Vector Mean:** `{res['overall_mean']:.4f}` | **Std:** `{res['overall_std']:.4f}`")

            if res['records']:
                df = pd.DataFrame(res['records'])
                st.dataframe(df, use_container_width=True)

# ==========================================
# 8. 📈 MONITORING DASHBOARD
# ==========================================
elif nav_option == "📈 Monitoring Dashboard":
    st.header("📈 Operational Telemetry & Latency Dashboard")
    st.markdown("Real-time telemetry tracking prediction counts, inference latency percentiles, and low-confidence rates.")

    stats = get_latency_stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", stats['total_predictions'])
    c2.metric("Avg Latency", f"{stats['avg_latency_ms']} ms")
    c3.metric("p95 Latency", f"{stats['p95_latency_ms']} ms")
    c4.metric("Low-Confidence Rate", f"{stats['low_confidence_rate_pct']}%")

    if stats['class_distribution']:
        st.subheader("Prediction Distribution by Class")
        st.bar_chart(stats['class_distribution'])

    st.subheader("Recent Inference Audit Logs")
    logs = get_recent_audit_logs(limit=20)
    if logs:
        st.dataframe(pd.DataFrame(logs), use_container_width=True)
    else:
        st.info("No inference logs recorded yet. Run predictions in the prediction tab.")

# ==========================================
# 9. 📉 DRIFT MONITORING
# ==========================================
elif nav_option == "📉 Drift Monitoring":
    st.header("📉 Input Distribution Data Drift Detection")
    st.markdown("Statistical comparison of incoming inference CT image intensity profiles against training baseline.")

    uploaded_file = st.file_uploader("Upload Image to Test Data Drift:", type=["png", "jpg", "jpeg"], key="drift_upload")
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        from src.data.preprocessing import load_and_preprocess_image
        tensor, _ = load_and_preprocess_image(image)

        stats = compute_image_stats(tensor)
        drift = detect_data_drift(stats)

        st.subheader(f"Drift Status: `{drift['drift_status']}`")
        st.write(f"**Drift Score:** `{drift['drift_score']}`")
        st.write(f"**Details:** {drift['message']}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Current Sample Stats:**")
            st.json(stats)
        with col2:
            st.markdown("**Training Baseline Stats:**")
            st.json(drift['reference_baseline'])

        st.caption(f"ℹ️ {drift['disclaimer']}")

# ==========================================
# 10. 📝 PERFORMANCE FEEDBACK
# ==========================================
elif nav_option == "📝 Performance Feedback":
    st.header("📝 Verified Ground-Truth Feedback Loop")
    st.markdown("Clinical researchers can log verified ground truth labels to track post-deployment model accuracy.")

    with st.form("feedback_form"):
        pred_id = st.text_input("Prediction ID:", value="pred-example-101")
        verified_cls = st.selectbox("Verified True Class:", list(CLASS_LABELS.values()))
        predicted_cls = st.selectbox("Model Predicted Class:", list(CLASS_LABELS.values()))
        submit_fb = st.form_submit_button("Submit Verified Feedback")

    if submit_fb:
        success, msg = log_ground_truth_feedback(pred_id, verified_cls, predicted_cls)
        if success:
            st.success("Feedback recorded successfully!")
        else:
            st.error(msg)

    st.divider()
    st.subheader("Verified Performance Summary")
    perf = get_verified_performance_metrics()
    st.info(perf['notice'])
    if perf['has_verified_data']:
        st.json(perf['metrics'])
    else:
        st.write(f"Total Verified Feedback Entries: `{perf['verified_count']}`")

# ==========================================
# 11. 🛡️ RESPONSIBLE AI
# ==========================================
elif nav_option == "🛡️ Responsible AI":
    st.header("🛡️ Responsible AI, Transparency & Ethics")

    st.markdown("""
    ### 1. Explainability & Interpretability
    - **Grad-CAM Visualizations:** Heatmaps highlight anatomical regions driving neural network decisions.
    
    ### 2. Human-in-the-Loop Oversight
    - **Low-Confidence Triggers:** Predictions with confidence below **60.0%** trigger automated warnings recommending expert human review.

    ### 3. Privacy & Data Handling
    - **No Patient Image Retention:** Uploaded CT images are processed in-memory and deleted immediately after inference. Only non-PII numerical metadata is recorded in logs.

    ### 4. Limitations & Scope
    - **Educational / Research Prototype Only:** Not cleared by the FDA or intended for clinical diagnosis.
    - **Dataset Boundaries:** Evaluated on public benchmark CT scan splits. Performance may vary on different scanner modalities.

    ### 5. Auditability
    - All inference operations produce structured JSON audit logs containing model version, timestamp, latency, and confidence metrics.
    """)

# ==========================================
# 12. 🩺 SYSTEM HEALTH
# ==========================================
elif nav_option == "🩺 System Health":
    st.header("🩺 Operational System Health & Diagnostics")

    health = check_system_health()
    st.subheader(f"Overall Status: `{health['status']}`")

    st.json(health['checks'])

st.divider()
st.caption("Explainable Lung Cancer Classification | Optum AI/ML Engineer Aligned Portfolio Project | Educational & Research Purpose Only")
