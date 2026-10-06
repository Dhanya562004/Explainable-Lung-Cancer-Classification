import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input

from utils import (
    IMAGE_SIZE, CLASS_LABELS, CLASS_NAMES, DEFAULT_MODEL_PATH
)

_cached_model = None

def get_model(model_path=DEFAULT_MODEL_PATH):
    """
    Loads and caches trained Keras model.
    """
    global _cached_model
    if _cached_model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
        _cached_model = tf.keras.models.load_model(model_path)
    return _cached_model

def preprocess_image_for_prediction(img_input, target_size=IMAGE_SIZE):
    """
    Preprocesses PIL image, image path, or numpy array into model input tensor.
    """
    if isinstance(img_input, str):
        if not os.path.exists(img_input):
            raise FileNotFoundError(f"Input image file not found: {img_input}")
        img = Image.open(img_input).convert('RGB')
    elif isinstance(img_input, Image.Image):
        img = img_input.convert('RGB')
    elif isinstance(img_input, np.ndarray):
        img = Image.fromarray(img_input).convert('RGB')
    else:
        raise ValueError("Unsupported image input format. Expected file path, PIL Image, or numpy array.")

    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    return img_preprocessed, img

def predict_ct_scan(img_input, model_path=DEFAULT_MODEL_PATH, model=None):
    """
    Predicts lung CT image classification class, confidence, and full probability breakdown.
    
    Returns:
        dict containing:
            - 'predicted_class': Display string of top predicted class
            - 'confidence': Prediction confidence percentage (float 0-100)
            - 'probabilities': Dict of class names to confidence percentages
            - 'predicted_index': Integer index of predicted class
            - 'original_image': PIL Image instance
            - 'preprocessed_tensor': Preprocessed Keras input array
    """
    if model is None:
        model = get_model(model_path)

    img_tensor, original_img = preprocess_image_for_prediction(img_input)

    raw_preds = model.predict(img_tensor, verbose=0)[0]
    predicted_idx = int(np.argmax(raw_preds))
    confidence_pct = float(raw_preds[predicted_idx] * 100.0)

    probabilities = {
        CLASS_LABELS[i]: float(raw_preds[i] * 100.0)
        for i in range(len(CLASS_NAMES))
    }

    return {
        'predicted_class': CLASS_LABELS[predicted_idx],
        'confidence': confidence_pct,
        'probabilities': probabilities,
        'predicted_index': predicted_idx,
        'original_image': original_img,
        'preprocessed_tensor': img_tensor
    }

if __name__ == '__main__':
    import sys
    test_img = sys.argv[1] if len(sys.argv) > 1 else None
    if test_img:
        res = predict_ct_scan(test_img)
        print("Prediction Result:")
        print(f"Predicted Class: {res['predicted_class']}")
        print(f"Confidence:      {res['confidence']:.2f}%")
        print("Probabilities:")
        for cls, prob in res['probabilities'].items():
            print(f"  {cls}: {prob:.2f}%")
    else:
        print("Usage: python predict.py <path_to_ct_image>")
