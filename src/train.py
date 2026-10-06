import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from utils import (
    IMAGE_SIZE, BATCH_SIZE, NUM_CLASSES,
    DEFAULT_DATASET_DIR, DEFAULT_MODEL_PATH, DEFAULT_RESULTS_DIR,
    get_data_generators
)

def build_model(input_shape=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES):
    """
    Constructs transfer learning model based on Xception backbone
    with an enhanced classification head.
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

def plot_training_curves(history_stage1, history_stage2, save_path):
    """
    Plots and saves training vs validation accuracy and loss curves across both training stages.
    """
    acc = history_stage1.history['accuracy'] + history_stage2.history['accuracy']
    val_acc = history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy']
    loss = history_stage1.history['loss'] + history_stage2.history['loss']
    val_loss = history_stage1.history['val_loss'] + history_stage2.history['val_loss']

    stage1_epochs = len(history_stage1.history['accuracy'])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', color='#1f77b4', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', color='#ff7f0e', linewidth=2)
    plt.axvline(x=stage1_epochs, color='gray', linestyle='--', label='Fine-tuning Start (Stage 2)')
    plt.title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', color='#1f77b4', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', color='#ff7f0e', linewidth=2)
    plt.axvline(x=stage1_epochs, color='gray', linestyle='--', label='Fine-tuning Start (Stage 2)')
    plt.title('Training & Validation Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved training curves to: {save_path}")

def train_model():
    """
    Executes two-stage transfer learning fine-tuning pipeline.
    """
    os.makedirs(os.path.dirname(DEFAULT_MODEL_PATH), exist_ok=True)
    os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)

    print("[INFO] Preparing dataset generators...")
    train_gen, valid_gen, test_gen = get_data_generators()

    model, base_model = build_model()
    model.summary()

    # ==========================================
    # STAGE 1: Feature Extraction (Frozen Base)
    # ==========================================
    print("\n" + "="*50)
    print("[STAGE 1] Training classification head with frozen Xception base...")
    print("="*50)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_stage1 = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(DEFAULT_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    history_stage1 = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=20,
        callbacks=callbacks_stage1
    )

    # ==========================================
    # STAGE 2: Fine-Tuning (Unfreeze Top Layers)
    # ==========================================
    print("\n" + "="*50)
    print("[STAGE 2] Unfreezing top 30 layers of Xception base for fine-tuning...")
    print("="*50)

    base_model.trainable = True
    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    print(f"[INFO] Total layers in Xception base: {len(base_model.layers)}")
    print(f"[INFO] Trainable layers count: {sum([1 for l in model.layers if l.trainable])}")

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_stage2 = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(DEFAULT_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
    ]

    history_stage2 = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=25,
        callbacks=callbacks_stage2
    )

    # Save final curves
    curves_path = os.path.join(DEFAULT_RESULTS_DIR, 'training_curves.png')
    plot_training_curves(history_stage1, history_stage2, curves_path)
    print("\n[SUCCESS] Two-stage model training complete. Best model saved to:", DEFAULT_MODEL_PATH)

if __name__ == '__main__':
    train_model()
