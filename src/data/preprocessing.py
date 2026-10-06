import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input

from src.config import (
    IMAGE_SIZE, BATCH_SIZE, CLASS_NAMES, DEFAULT_DATASET_DIR
)

def validate_image_file(img_input):
    """
    Validates if input is a valid image file, PIL Image, or numpy array.
    """
    if isinstance(img_input, str):
        if not os.path.exists(img_input):
            return False, f"File not found: {img_input}"
        try:
            with Image.open(img_input) as img:
                img.verify()
            return True, "Valid image path"
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
    elif isinstance(img_input, Image.Image):
        return True, "Valid PIL Image instance"
    elif isinstance(img_input, np.ndarray):
        if img_input.ndim in (2, 3):
            return True, "Valid numpy array"
        return False, f"Invalid numpy array shape: {img_input.shape}"
    else:
        return False, f"Unsupported input type: {type(img_input)}"

def load_and_preprocess_image(img_input, target_size=IMAGE_SIZE):
    """
    Preprocesses PIL Image, image path, or numpy array into model input tensor.
    
    1. Loads / converts image to RGB
    2. Resizes to target_size (default 299x299 for Xception)
    3. Converts to float32 numpy array
    4. Expands batch dimension -> shape (1, 299, 299, 3)
    5. Normalizes using Xception preprocess_input [-1, 1] range.

    Returns:
        tuple: (preprocessed_tensor, original_pil_image)
    """
    is_valid, err_msg = validate_image_file(img_input)
    if not is_valid:
        raise ValueError(f"Image preprocessing failed: {err_msg}")

    if isinstance(img_input, str):
        img = Image.open(img_input).convert('RGB')
    elif isinstance(img_input, Image.Image):
        img = img_input.convert('RGB')
    elif isinstance(img_input, np.ndarray):
        img = Image.fromarray(img_input).convert('RGB')

    img_resized = img.resize(target_size)
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    return img_preprocessed, img

def get_data_generators(dataset_dir=DEFAULT_DATASET_DIR, batch_size=BATCH_SIZE, target_size=IMAGE_SIZE):
    """
    Creates and returns Keras ImageDataGenerators for train, validation, and test splits.
    
    Distinct pipelines:
    - Training pipeline: Moderate data augmentation (rotation, translation, zoom, horizontal flip)
      to prevent overfitting without distorting medical characteristics.
    - Validation / Test pipelines: Isolated preprocessing without augmentation for accurate evaluation.
    """
    train_dir = os.path.join(dataset_dir, 'train')
    valid_dir = os.path.join(dataset_dir, 'valid')
    test_dir = os.path.join(dataset_dir, 'test')

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    eval_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        classes=CLASS_NAMES
    ) if os.path.exists(train_dir) else None

    valid_gen = eval_datagen.flow_from_directory(
        valid_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        classes=CLASS_NAMES
    ) if os.path.exists(valid_dir) else None

    test_gen = eval_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        classes=CLASS_NAMES
    ) if os.path.exists(test_dir) else None

    return train_gen, valid_gen, test_gen
