import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model

from src.config import INPUT_SHAPE, NUM_CLASSES

def build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Constructs transfer learning classification model based on Xception backbone
    with a 2-stage fine-tunable custom top head.

    Head architecture:
      - GlobalAveragePooling2D
      - BatchNormalization
      - Dropout (0.4)
      - Dense (128 units, ReLU activation)
      - BatchNormalization
      - Dropout (0.3)
      - Dense (num_classes, Softmax activation)
    """
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base model initially
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = BatchNormalization(name='head_batchnorm_1')(x)
    x = Dropout(0.4, name='head_dropout_1')(x)
    x = Dense(128, activation='relu', name='head_dense_1')(x)
    x = BatchNormalization(name='head_batchnorm_2')(x)
    x = Dropout(0.3, name='head_dropout_2')(x)
    outputs = Dense(num_classes, activation='softmax', name='classification_head')(x)

    model = Model(inputs=base_model.input, outputs=outputs, name='Xception_Lung_Cancer_Classifier')
    return model, base_model
