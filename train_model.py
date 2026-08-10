import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import json
import os

print("TensorFlow version:", tf.__version__)
print("Starting MobileNetV2 training setup...")

# ── Configuration ──────────────────────────────────────────────
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 20
DATASET_DIR = "dataset_split"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# ── Check dataset exists ───────────────────────────────────────
if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: Dataset split folder not found at {TRAIN_DIR}")
    print("Please run split_dataset.py first.")
    exit(1)

# ── Data preparation ───────────────────────────────────────────
print("\nPreparing data generators...")

# Use mobilenet_v2.preprocess_input instead of rescale=1./255
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Only preprocess for val and test, no augmentation
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")

# Save class names mapping
class_indices = train_generator.class_indices
class_names = {str(v): k for k, v in class_indices.items()}
with open('class_names.json', 'w') as f:
    json.dump(class_names, f, indent=2)
print(f"Class names saved: {class_names}")

# ── Build model ────────────────────────────────────────────────
print("\nBuilding MobileNetV2 model...")

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model built successfully!")
print(f"Total parameters: {model.count_params():,}")

# ── Phase 1 Training ───────────────────────────────────────────
print("\n=== PHASE 1: Feature Extraction Training ===")

callbacks_phase1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model_phase1.h5',
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    )
]

history1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=val_generator,
    callbacks=callbacks_phase1,
    verbose=1
)

phase1_acc = max(history1.history['val_accuracy'])
print(f"\nPhase 1 Best Validation Accuracy: {phase1_acc * 100:.2f}%")

# ── Phase 2 Fine Tuning ────────────────────────────────────────
print("\n=== PHASE 2: Fine Tuning ===")

base_model.trainable = True
# Freeze the first 100 layers and unfreeze the rest
for layer in base_model.layers[:100]:
    layer.trainable = False

trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
print(f"Trainable layers in base model: {trainable_count}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'dental_ai_model.h5',
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    )
]

history2 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE2,
    validation_data=val_generator,
    callbacks=callbacks_phase2,
    verbose=1
)

# ── Final evaluation on TEST SET ───────────────────────────────
print("\n=== FINAL EVALUATION ON UNSEEN TEST SET ===")
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
print(f"Final Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

# ── Save final model ───────────────────────────────────────────
model.save('dental_ai_model.h5')
print("\nModel saved as: dental_ai_model.h5")
print("\nTraining Complete!")