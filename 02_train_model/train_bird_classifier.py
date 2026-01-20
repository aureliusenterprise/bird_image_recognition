# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 20:02:12 2025

@author: AndreasWombacher
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pathlib

# ================================
# SETTINGS
# ================================
base_dir = 'jonathan/'
TRAIN_DIR = base_dir+"dataset/train"
VAL_DIR   = base_dir+"dataset/val"

CROP_LEFT   = 130
CROP_RIGHT  = 185
CROP_BOTTOM = 150

ORIG_W = 640
ORIG_H = 480

CROPPED_W = ORIG_W - CROP_LEFT - CROP_RIGHT
CROPPED_H = ORIG_H - CROP_BOTTOM

TARGET_SIZE = (160, 160)
BATCH_SIZE = 64
EPOCHS = 20

# OUTPUT_MODEL = base_dir+"output/bird_classifier.h5"
# OUTPUT_TFLITE = base_dir+"output/bird_classifier_int8.tflite"

OUTPUT_MODEL = "bird_classifier.h5"
OUTPUT_TFLITE = "bird_classifier_int8.tflite"

# ================================
# LOAD DATA
# ================================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(ORIG_H, ORIG_W),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    image_size=(ORIG_H, ORIG_W),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE

# ================================
# CROPPING LAYER
# ================================
crop_layer = layers.Cropping2D(
    cropping=((0, CROP_BOTTOM), (CROP_LEFT, CROP_RIGHT))
)

# ================================
# AUGMENTATION
# ================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
])

# ================================
# BUILD MODEL
# ================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=TARGET_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = layers.Input(shape=(ORIG_H, ORIG_W, 3))

x = crop_layer(inputs)
x = layers.Resizing(*TARGET_SIZE)(x)
x = data_augmentation(x)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)  # binary classifier

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================================
# TRAIN
# ================================
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

model.save(OUTPUT_MODEL)
print("Saved:", OUTPUT_MODEL)

# ================================
# FINE-TUNE (optional)
# ================================
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=5)

model.save(OUTPUT_MODEL)
print("Saved fine-tuned model")

# ================================
# TFLITE CONVERSION (INT8)
# ================================

#model = tf.keras.models.load_model(OUTPUT_MODEL)  # or your in-memory model

# ================================
# 2. Define representative dataset
#    Needed for INT8 quantization
# ================================
def representative_dataset():
    # Use a small sample of your training dataset
    # train_ds should be your tf.data.Dataset of images
    for images, _ in train_ds.take(100):
        yield [tf.cast(images, tf.float32)]

# ================================
# 3. Create TFLite converter
# ================================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]       # enable quantization
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# ================================
# 4. Convert to TFLite
# ================================
tflite_model = converter.convert()

# ================================
# 5. Save TFLite model
# ================================
with open("bird_classifier_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite INT8 model saved: bird_classifier_int8.tflite")
#%%
# def representative_dataset():
#     for images, _ in train_ds.take(100):
#         yield [tf.cast(images, tf.float32)]

# converter = tf.lite.TFLiteConverter.from_saved_model(OUTPUT_MODEL)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.representative_dataset = representative_dataset
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8

# tflite_model = converter.convert()

# with open(OUTPUT_TFLITE, "wb") as f:
#     f.write(tflite_model)

# print("Saved:", OUTPUT_TFLITE)

#%%
# Evaluate Keras model
loss, accuracy = model.evaluate(val_ds)
print(f"Validation loss: {loss:.4f}")
print(f"Validation accuracy: {accuracy:.4f}")

#%%
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    preds = (preds > 0.5).astype(int).flatten()  # threshold 0.5
    y_pred.extend(preds)
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Detailed classification report
report = classification_report(y_true, y_pred, target_names=["not_bird", "bird"])
print(report)

#%%

