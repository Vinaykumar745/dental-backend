import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import json
import os

print("TensorFlow version:", tf.__version__)
print("Starting training setup...")

# ── Configuration ──────────────────────────────────────────────
IMG_SIZE = 300
BATCH_SIZE = 16
EPOCHS_PHASE1 = 30
EPOCHS_PHASE2 = 25
DATASET_PATH = "dataset"

# ── Check dataset exists ───────────────────────────────────────
if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Dataset folder not found at {DATASET_PATH}")
    print("Please create the dataset folder with subfolders for each disease")
    exit(1)

classes = sorted(os.listdir(DATASET_PATH))
print(f"Found {len(classes)} classes: {classes}")

total_images = 0
for cls in classes:
    class_path = os.path.join(DATASET_PATH, cls)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  {cls}: {count} images")
        total_images += count

print(f"Total images: {total_images}")

if total_images < 50:
    print("WARNING: Very few images. Need at least 200+ for good accuracy.")

# ── Data preparation ───────────────────────────────────────────
print("\nPreparing data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    shear_range=0.2,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# Save class names mapping
class_indices = train_generator.class_indices
class_names = {str(v): k for k, v in class_indices.items()}
with open('class_names.json', 'w') as f:
    json.dump(class_names, f, indent=2)
print(f"Class names saved: {class_names}")

# ── Build model ────────────────────────────────────────────────
print("\nBuilding model...")

base_model = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
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
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
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

# ── Final evaluation ───────────────────────────────────────────
print("\n=== FINAL EVALUATION ===")
val_loss, val_acc = model.evaluate(val_generator, verbose=1)
print(f"Final Validation Loss: {val_loss:.4f}")
print(f"Final Validation Accuracy: {val_acc * 100:.2f}%")

# ── Save final model ───────────────────────────────────────────
model.save('dental_ai_model.h5')
print("\nModel saved as: dental_ai_model.h5")
print("Class names saved as: class_names.json")
print("\nTraining Complete!")
print(f"Your model can detect {num_classes} classes with {val_acc * 100:.2f}% accuracy")