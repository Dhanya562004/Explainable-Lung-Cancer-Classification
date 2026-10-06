import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input

# Project Configuration Constants
IMAGE_SIZE = (299, 299)
BATCH_SIZE = 16
NUM_CLASSES = 4

# Standardized Class Mappings
CLASS_LABELS = {
    0: 'Adenocarcinoma',
    1: 'Large Cell Carcinoma',
    2: 'Normal',
    3: 'Squamous Cell Carcinoma'
}

CLASS_NAMES = ['adenocarcinoma', 'large_cell_carcinoma', 'normal', 'squamous_cell_carcinoma']

DEFAULT_DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.keras'))
DEFAULT_RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

def get_data_generators(dataset_dir=DEFAULT_DATASET_DIR, batch_size=BATCH_SIZE, target_size=IMAGE_SIZE):
    """
    Creates and returns Keras ImageDataGenerators for train, validation, and test splits.
    Applies moderate, medically appropriate augmentation to the training set only.
    Validation and test sets use isolated preprocessing without data augmentation.
    """
    train_dir = os.path.join(dataset_dir, 'train')
    valid_dir = os.path.join(dataset_dir, 'valid')
    test_dir = os.path.join(dataset_dir, 'test')

    # Training generator with moderate augmentation suitable for chest CT scans
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation and test generators without augmentation
    eval_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        classes=CLASS_NAMES
    )

    valid_generator = eval_datagen.flow_from_directory(
        valid_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        classes=CLASS_NAMES
    )

    test_generator = eval_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        classes=CLASS_NAMES
    )

    return train_generator, valid_generator, test_generator
