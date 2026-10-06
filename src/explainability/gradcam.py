import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

from src.config import (
    DEFAULT_MODEL_PATH, DEFAULT_RESULTS_DIR
)
from src.models.inference import predict_ct_scan, get_model
from src.data.preprocessing import load_and_preprocess_image

def find_target_conv_layer(model):
    """
    Finds the name of the final convolutional layer in the Xception architecture for Grad-CAM.
    """
    target_names = ['block14_sepconv2_act', 'block14_sepconv2', 'conv2d_4']
    for name in target_names:
        try:
            model.get_layer(name)
            return name
        except ValueError:
            pass

    # Search backwards for the last 4D output layer (Conv2D or SeparableConv2D)
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name

    raise ValueError("Could not find a valid 4D convolutional layer in the model for Grad-CAM.")

def make_gradcam_heatmap(img_tensor, model, last_conv_layer_name=None, pred_index=None):
    """
    Generates Grad-CAM heatmap array for input image tensor and target class index.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_target_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def generate_gradcam_overlay(img_input, model=None, model_path=DEFAULT_MODEL_PATH, alpha=0.4, pred_index=None):
    """
    Full pipeline to generate Grad-CAM visualization for a given CT image input.

    Returns:
        dict containing:
            - 'superimposed_image': RGB uint8 numpy array of overlay
            - 'heatmap_colored': RGB uint8 numpy array of jet colormap heatmap
            - 'predicted_class': String label of predicted class
            - 'confidence': Prediction confidence percentage
            - 'confidence_level': 'HIGH', 'MEDIUM', or 'LOW'
            - 'probabilities': Class probability dictionary
            - 'original_image': Original uint8 numpy array
    """
    if model is None:
        model = get_model(model_path)

    pred_res = predict_ct_scan(img_input, model=model)
    img_tensor = pred_res['preprocessed_tensor']
    orig_pil = pred_res['original_image']

    if pred_index is None:
        pred_index = pred_res['predicted_index']

    heatmap = make_gradcam_heatmap(img_tensor, model, pred_index=pred_index)

    orig_np = np.array(orig_pil)
    if orig_np.ndim == 2:
        orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)
    
    orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
    h, w = orig_np.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    superimposed_bgr = cv2.addWeighted(orig_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)
    superimposed_rgb = cv2.cvtColor(superimposed_bgr, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    return {
        'prediction_id': pred_res['prediction_id'],
        'superimposed_image': superimposed_rgb,
        'heatmap_colored': heatmap_rgb,
        'heatmap_raw': heatmap_resized,
        'predicted_class': pred_res['predicted_class'],
        'confidence': pred_res['confidence'],
        'confidence_level': pred_res['confidence_level'],
        'probabilities': pred_res['probabilities'],
        'requires_human_review': pred_res['requires_human_review'],
        'recommendation': pred_res['recommendation'],
        'model_version': pred_res['model_version'],
        'inference_latency_ms': pred_res['inference_latency_ms'],
        'original_image': orig_np
    }

def save_sample_gradcam(sample_img_path, save_path=None, model_path=DEFAULT_MODEL_PATH):
    """
    Saves side-by-side visual comparison figure of Original CT Scan, Grad-CAM Heatmap, and Superimposed Overlay.
    """
    if save_path is None:
        save_path = os.path.join(DEFAULT_RESULTS_DIR, 'gradcam_sample.png')

    res = generate_gradcam_overlay(sample_img_path, model_path=model_path)

    plt.figure(figsize=(14, 4.5))

    plt.subplot(1, 3, 1)
    plt.imshow(res['original_image'])
    plt.title('Original CT Scan', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(res['heatmap_colored'])
    plt.title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(res['superimposed_image'])
    plt.title(f"Overlay: {res['predicted_class']} ({res['confidence']:.1f}%)", fontsize=12, fontweight='bold')
    plt.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved Grad-CAM sample visualization to: {save_path}")

if __name__ == '__main__':
    import sys
    test_img = sys.argv[1] if len(sys.argv) > 1 else None
    if test_img:
        save_sample_gradcam(test_img)
