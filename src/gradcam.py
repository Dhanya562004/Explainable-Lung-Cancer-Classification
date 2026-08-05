import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

from utils import (
    IMAGE_SIZE, DEFAULT_MODEL_PATH, DEFAULT_RESULTS_DIR
)
from predict import predict_ct_scan, get_model, preprocess_image_for_prediction

def find_target_conv_layer(model):
    """
    Finds the name of the final convolutional layer in the Xception architecture.
    """
    # Standard layer name for Xception base final activation
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
    Generates Grad-CAM heatmap array for input image tensor and target class.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_target_conv_layer(model)

    # Sub-model that returns target conv layer activation and final predictions
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    # Gradient of target class score with respect to feature map
    grads = tape.gradient(loss, conv_outputs)

    # Vector of mean gradient intensity per feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to keep only features that positively contribute to class prediction
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
            - 'probabilities': Class probability dictionary
    """
    if model is None:
        model = get_model(model_path)

    pred_res = predict_ct_scan(img_input, model=model)
    img_tensor = pred_res['preprocessed_tensor']
    orig_pil = pred_res['original_image']

    if pred_index is None:
        pred_index = pred_res['predicted_index']

    heatmap = make_gradcam_heatmap(img_tensor, model, pred_index=pred_index)

    # Convert original image to BGR numpy array for OpenCV processing
    orig_np = np.array(orig_pil)
    orig_bgr = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
    h, w = orig_np.shape[:2]

    # Resize heatmap to match input image dimensions
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply Jet color map
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Superimpose heatmap on original CT image
    superimposed_bgr = cv2.addWeighted(orig_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)
    superimposed_rgb = cv2.cvtColor(superimposed_bgr, cv2.COLOR_BGR2RGB)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    return {
        'superimposed_image': superimposed_rgb,
        'heatmap_colored': heatmap_rgb,
        'heatmap_raw': heatmap_resized,
        'predicted_class': pred_res['predicted_class'],
        'confidence': pred_res['confidence'],
        'probabilities': pred_res['probabilities'],
        'original_image': orig_np
    }

def save_sample_gradcam(sample_img_path, save_path=None, model_path=DEFAULT_MODEL_PATH):
    """
    Saves a side-by-side visualization of Original CT Image, Grad-CAM Heatmap, and Superimposed Overlay.
    """
    if save_path is None:
        save_path = os.path.join(DEFAULT_RESULTS_DIR, 'gradcam_sample.png')

    res = generate_gradcam_overlay(sample_img_path, model_path=model_path)

    plt.figure(figsize=(14, 4.5))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(res['original_image'])
    plt.title('Original CT Scan', fontsize=12, fontweight='bold')
    plt.axis('off')

    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(res['heatmap_colored'])
    plt.title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    plt.axis('off')

    # Superimposed Overlay
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
    else:
        print("Usage: python gradcam.py <path_to_ct_image>")
