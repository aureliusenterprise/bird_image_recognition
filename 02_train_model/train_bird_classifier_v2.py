# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 21:32:38 2026

@author: AndreasWombacher
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pathlib
import matplotlib.pyplot as plt

# ================================
# SETTINGS
# ================================
base_dir = 'jonathan/'
TRAIN_DIR = base_dir+"dataset/train"
VAL_DIR   = base_dir+"dataset/val"

CROP_LEFT = 115
CROP_RIGHT = 205
CROP_BOTTOM = 110
CROP_TOP = 50

ORIG_W = 640
ORIG_H = 480

CROPPED_W = ORIG_W - CROP_LEFT - CROP_RIGHT
CROPPED_H = ORIG_H - CROP_BOTTOM

TARGET_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 25

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
# crop_layer = layers.Cropping2D(
#     cropping=((0, CROP_BOTTOM), (CROP_LEFT, CROP_RIGHT))
# )

# ================================
# AUGMENTATION
# ================================
data_augmentation = tf.keras.Sequential([
    #layers.Input(shape=(160,160,3)),
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.2),
    layers.RandomRotation(0.05),
    layers.RandomContrast(0.2),
])

# ================================
# BUILD MODEL (no preprocess_input inside)
# ================================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=TARGET_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = layers.Input(shape=(160, 160, 3))

# x = crop_layer(inputs)
# x = layers.Resizing(*TARGET_SIZE)(x)
x = data_augmentation(inputs)
# --- REMOVE preprocess_input here ---
# x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

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
def preprocess_dataset(dataset):
    def preprocess(image, label):
        img_cropped = tf.image.crop_to_bounding_box(
            image,
            offset_height=CROP_TOP,
            offset_width=CROP_LEFT,
            target_height=ORIG_H - CROP_BOTTOM - CROP_TOP,
            target_width=ORIG_W - CROP_RIGHT - CROP_LEFT
        )
        #img_resized = img_cropped
        img_resized = tf.image.resize(img_cropped, TARGET_SIZE)
        img_resized = tf.cast(img_resized, tf.float32)
        img_resized = (img_resized / 127.5) - 1.0
        return img_resized, label
    return dataset.map(preprocess, num_parallel_calls=AUTOTUNE)

train_ds_proc = preprocess_dataset(train_ds).prefetch(AUTOTUNE)
val_ds_proc = preprocess_dataset(val_ds).prefetch(AUTOTUNE)

#%%
# check the created dataset, whether it has the right transformations
images, labels = next(iter(train_ds_proc))
#images, labels = next(iter(train_ds))

# Take the first image from the batch
img = images[0]

# If your pipeline normalized to [-1, 1] (e.g., MobileNet preprocessing),
# convert back to [0, 1] for display
img_disp = (img + 1.0) / 2.0

plt.imshow(img_disp)
#plt.imshow(img)
plt.title(f"Label: {labels[0].numpy()}")
plt.axis('off')
plt.show()
#%%

class_weight = {
    0: 1.0,   # NON-bird
    1: 2.0    # bird
}

history = model.fit(
    train_ds_proc,
    validation_data=val_ds_proc,
    epochs=EPOCHS,
    class_weight=class_weight
)

model.save(OUTPUT_MODEL, save_format="h5")
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

model.fit(train_ds_proc, validation_data=val_ds_proc, epochs=5)

model.save(OUTPUT_MODEL, save_format="h5")
print("Saved fine-tuned model")

# ================================
# TFLITE CONVERSION (INT8)
# ================================
def representative_dataset():
    for images, _ in train_ds_proc.take(100):
        yield [tf.cast(images, tf.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()

with open(OUTPUT_TFLITE, "wb") as f:
    f.write(tflite_model)

print("✅ TFLite INT8 model saved:", OUTPUT_TFLITE)

#%%
# Evaluate Keras model

def preprocess_val(image, label):
 def preprocess(image, label):
     img_cropped = tf.image.crop_to_bounding_box(
         image,
         offset_height=CROP_TOP,
         offset_width=CROP_LEFT,
         target_height=ORIG_H - CROP_BOTTOM - CROP_TOP,
         target_width=ORIG_W - CROP_RIGHT - CROP_LEFT
     )
     #img_resized = img_cropped
     img_resized = tf.image.resize(img_cropped, TARGET_SIZE)
     img_resized = tf.cast(img_resized, tf.float32)
     img_resized = (img_resized / 127.5) - 1.0
     return img_resized, label
    
val_ds_proc = val_ds.map(preprocess_val).prefetch(tf.data.AUTOTUNE)

loss, accuracy = model.evaluate(val_ds_proc)
print(f"Validation loss: {loss:.4f}")
print(f"Validation accuracy: {accuracy:.4f}")


#%%
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

y_true = []
y_pred = []

for images, labels in val_ds_proc:
    preds = model.predict(images)
    preds = (preds > 0.1).astype(int).flatten()  # threshold 0.5
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
