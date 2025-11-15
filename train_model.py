import tensorflow as tf
# from tensorflow import keras # Removed - Keras is accessed via tf.keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json

# Configuration
IMG_SIZE = 224  # MobileNetV2 expects 224x224 images
BATCH_SIZE = 32
EPOCHS = 15  # Adjust based on time - can reduce to 10 if needed
DATASET_PATH = 'dataset'  # Your organized dataset folder
MODEL_SAVE_PATH = 'model/isl_model.h5'
LABELS_SAVE_PATH = 'model/labels.json'

print("=" * 50)
print("ISL SIGN LANGUAGE MODEL TRAINING")
print("=" * 50)

# Step 1: Load and prepare dataset
print("\n[1/6] Loading dataset...")
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80% train, 20% validation
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Training data
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation data
validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Get class labels
class_labels = {v: k for k, v in train_generator.class_indices.items()}
num_classes = len(class_labels)

print(f"✓ Dataset loaded successfully!")
print(f"  - Number of classes: {num_classes}")
print(f"  - Training samples: {train_generator.samples}")
print(f"  - Validation samples: {validation_generator.samples}")
print(f"  - Classes: {list(class_labels.values())}")

# Save class labels for later use
os.makedirs('model', exist_ok=True)
with open(LABELS_SAVE_PATH, 'w') as f:
    json.dump(class_labels, f)
print(f"✓ Class labels saved to {LABELS_SAVE_PATH}")

# Step 2: Build model with Transfer Learning
print("\n[2/6] Building MobileNetV2 model with transfer learning...")

# Load pre-trained MobileNetV2 (without top classification layer)
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model layers (we'll only train the top layers)
base_model.trainable = False

# Build model
model = tf.keras.Sequential([ # Changed from keras.Sequential
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Changed from keras.optimizers.Adam
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model architecture created!")
model.summary()

# Step 3: Train the model
print("\n[3/6] Starting training...")
print(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")
print("This may take 30-60 minutes depending on your hardware...\n")

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping( # Changed from keras.callbacks.EarlyStopping
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau( # Changed from keras.callbacks.ReduceLROnPlateau
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001
)

# Train
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\n✓ Training completed!")

# Step 4: Fine-tuning (optional but improves accuracy)
print("\n[4/6] Fine-tuning model (unfreezing some base layers)...")

# Unfreeze the last 20 layers of base model
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Changed from keras.optimizers.Adam
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continue training
fine_tune_epochs = 5
history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=fine_tune_epochs,
    callbacks=[early_stopping],
    verbose=1
)

print("✓ Fine-tuning completed!")

# Step 5: Evaluate model
print("\n[5/6] Evaluating model...")
val_loss, val_accuracy = model.evaluate(validation_generator, verbose=0)
print(f"✓ Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"✓ Validation Loss: {val_loss:.4f}")

# Step 6: Save model
print(f"\n[6/6] Saving model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("✓ Model saved successfully!")

# Plot training history
print("\nGenerating training plots...")
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('model/training_history.png')
print("✓ Training plots saved to model/training_history.png")

print("\n" + "=" * 50)
print("TRAINING COMPLETE!")
print("=" * 50)
print(f"Final Validation Accuracy: {val_accuracy * 100:.2f}%")
print(f"Model saved at: {MODEL_SAVE_PATH}")
print(f"Labels saved at: {LABELS_SAVE_PATH}")
print("\nNext step: Run the Flask backend (python backend/app.py)")
print("=" * 50)