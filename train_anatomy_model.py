import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2  # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
import json
import os

print("TensorFlow version:", tf.__version__)
print("Starting Anatomy Validation Model training...")

IMG_SIZE = 224 # MobileNetV2 standard
BATCH_SIZE = 16
EPOCHS = 10
DATASET_PATH = "../oral_dataset" # It's in the root folder

if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Dataset folder not found at {DATASET_PATH}")
    exit(1)

# Check classes
classes = sorted(os.listdir(DATASET_PATH))
print(f"Found {len(classes)} classes: {classes}")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
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

# Save class names mapping
class_indices = train_generator.class_indices
class_names = {str(v): k for k, v in class_indices.items()}
with open('anatomy_classes.json', 'w') as f:
    json.dump(class_names, f, indent=2)
print(f"Anatomy class names saved: {class_names}")

print("\nBuilding model...")
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n=== Training ===")
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    verbose=1
)

print("\nSaving anatomy model...")
model.save('anatomy_model.h5')
print("Anatomy model saved successfully as anatomy_model.h5")
