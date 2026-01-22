# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 21:37:05 2026

@author: AndreasWombacher
"""

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.cm as cm
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import json


CROP_LEFT = 115
CROP_RIGHT = 205
CROP_BOTTOM = 110
CROP_TOP = 50
TARGET_SIZE = (160,160)  # width, height
THRESHOLD = 0.1

ORIG_W = 640
ORIG_H = 480

OUTPUT_MODEL = "bird_classifier.h5"
base_dir = ""

#model = keras.models.load_model("model.h5")
model2 = keras.models.load_model(OUTPUT_MODEL)
model2.trainable = False
model2.summary()

#%%
def get_gradcam_heatmap(model, img_array, last_conv_layer_name="out_relu"):
    # 1. Access the sub-models
    # According to your summary, these are the layer names
    preprocessing_seq = model.get_layer("sequential")
    base_model = model.get_layer("mobilenetv2_1.00_160")
    
    # 2. Reconstruct the path to the internal target layer
    # We create a new sub-model for the base_model specifically
    # to get the internal convolutional output.
    base_internal_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )

    # 3. Record gradients through the entire stack
    with tf.GradientTape() as tape:
        # Pass input through preprocessing
        x = preprocessing_seq(img_array)
        # Pass through base model to get (conv_output, base_output)
        conv_outputs, base_output = base_internal_model(x)
        
        # Manually apply the remaining top layers of your 'functional_1' model
        # Following your summary order: GAP -> Dropout -> Dense
        x = model.get_layer("global_average_pooling2d")(base_output)
        x = model.get_layer("dropout")(x)
        predictions = model.get_layer("dense")(x)
        
        # Target the single output neuron
        loss = predictions[:, 0]

    # 4. Compute gradients and generate heatmap
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# Usage:
# target_layer = "out_relu" # Standard last layer for MobileNetV2
# heatmap = get_gradcam_heatmap(model, preprocessed_img, target_layer)

def display_gradcam(img, heatmap, alpha=0.4):
    # Rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    # Resize heatmap to match original image size
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))
    
    # Superimpose the heatmap on original image
    superimposed_img = jet * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    return superimposed_img

#frame_orig = cv2.imread(base_dir+"images/non/frame_00003524.jpg")

frame_orig = cv2.imread(base_dir+"images/frame_00000130.jpg")
frame_orig = cv2.imread(base_dir+"images/frame_00001841.jpg")
filename = "frame_00002097.jpg"
frame_orig = cv2.imread(f"{base_dir}images/{filename}")


frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)

# continue
def preprocess(image):
    h,w,_ = image.shape
    crop = image[
          CROP_TOP:max(0, h - CROP_BOTTOM),
          min(w, CROP_LEFT):max(0, w - CROP_RIGHT)
      ]
 
    # resize to model resolution
    img_resized = cv2.resize(crop, TARGET_SIZE)
    img_resized = (img_resized / 127.5) - 1.0
    return img_resized

img_np = preprocess(frame_orig)

#images, labels = next(iter(train_ds_proc))

# pick one image and make it a NumPy array
#img_np = images[0].numpy()      # shape (160,160,3)

# make a prediction
img_batch = np.expand_dims(img_np, axis=0) # Reshape to (1, 160, 160, 3)

# 2. Get the prediction
# In 2026, model.predict() is the standard for inference
prediction = model2.predict(img_batch)

# 3. Interpret the output
print(f"Raw model output: {prediction[0][0]}")

# If your model uses a sigmoid activation for binary classification:
if prediction[0][0] > 0.1:
    print("Predicted Class: Bird")
else:
    print("Predicted Class: Not Bird")

# add batch dimension
img = tf.expand_dims(img_np, axis=0)  # shape (1,160,160,3)

last_conv_layer_name = "out_relu"
heatmap = get_gradcam_heatmap(model2, img, last_conv_layer_name)

heatmap_img = display_gradcam(img_np, heatmap, alpha=0.02)

frame_disp2 = (img_np + 1.0) / 2.0
frame_disp2 = frame_disp2*255
frame_disp2 = frame_disp2.astype(int)
# Create a 1x2 grid of subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Show first image
axes[0].imshow(frame_disp2)
axes[0].axis("off")
axes[0].set_title("SourceImage")

# Show second image
axes[1].imshow(heatmap_img)
axes[1].axis("off")
axes[1].set_title("GradCAM heatmap")

# cbar = fig.colorbar(heatmap_img, ax=axes[1])
# cbar.set_label('Model Attention Intensity', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()

#%%

data = pd.read_csv(base_dir+"images/data.csv")

dd = data[data['filename']=='frame_    2097.jpg'].iloc[0]

contours = dd.contours

s = contours.replace("'",'"')
contours_json = json.loads(s)

img_rect = frame_orig.copy()
for item in contours_json:
    # Define start & end points (x, y)
    start_point = (item['x'], item['y'])      # top-left corner
    end_point = (item['x']+item['w'], item['y']+item['h'])      # bottom-right corner
    
    # Define color (B, G, R) and thickness
    color = (0, 255, 0)         # green
    thickness = 2               # -1 fills the rectangle
    
    # Draw rectangle
    cv2.rectangle(img_rect, start_point, end_point, color, thickness)

plt.imshow(img_rect)
plt.axis("off")
plt.show()

#%%
# look at the data: is it possible to distinguish different birds

data['len_contours'] = data['contours'].apply(lambda x: len(x))
data['contours_json'] = data['contours'].apply(lambda x: json.loads(x.replace("'",'"')))
data['len_contours_arr'] = data['contours_json'].apply(lambda x: len(x))
data['bird'] = data['probability'].apply(lambda x: x>0.1)
data['timestamp'] = pd.to_datetime(data['timestamp'])

data = data[data['timestamp']>'2026-01-04 08:00:00']

#%%
# plot frame timestamps

plt.scatter(data[data['bird']]['timestamp'],data[data['bird']]['probability'], color='green')
plt.scatter(data[~data['bird']]['timestamp'],data[~data['bird']]['probability'], color='red')
plt.xticks(rotation=45)   # 
plt.show()

#%%
# time difference in millisecnds between subsequent images
data['prev_timestamp'] = data['timestamp'].shift(1, fill_value=data.iloc[0]['timestamp'])
data['time_dif'] = data.apply(lambda x: (x['timestamp'] - x['prev_timestamp']).total_seconds()*1000, axis=1)
data['time_dif_s'] = data.apply(lambda x: (x['timestamp'] - x['prev_timestamp']).total_seconds()//1, axis=1)

plt.scatter(data[~data['bird']]['timestamp'],data[~data['bird']]['time_dif'], color='red', alpha=0.2)
plt.scatter(data[data['bird']]['timestamp'],data[data['bird']]['time_dif'], color='green', alpha=0.2)
plt.ylim(0,10000)
plt.xticks(rotation=45)   # 
plt.show()

data_agg = data.groupby(['bird','time_dif_s']).size().rename('cnt').reset_index()

data['row'] = range(len(data))

plt.scatter(data[data['bird']]['timestamp'],data[data['bird']]['row'], color='green', marker='X', alpha=0.1)
plt.scatter(data[~data['bird']]['timestamp'],data[~data['bird']]['row'], color='red', marker='.', alpha=0.05)
plt.xticks(rotation=45)   # 
plt.show()


#%%
# look at the location of the changed area
data_agg2 = data.groupby(['bird','len_contours_arr']).size().rename('cnt').reset_index()

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

X = data[data['len_contours_arr']==1]
X_ = pd.DataFrame(X['contours_json'].apply(lambda x: x[0]).to_list())

#scaler = StandardScaler()
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_)

pca = PCA(n_components=2)  # Reduce to 2 principal components for plotting
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

# If you have a categorical column for coloring points
pca_df["label"] = X["bird"].to_list()

plt.figure(figsize=(8,6))
plt.scatter(pca_df[pca_df['label']]["PC1"], pca_df[pca_df['label']]["PC2"], c='green', alpha = 0.05)  # or c=pca_df["label"] if you have categories
plt.scatter(pca_df[~pca_df['label']]["PC1"], pca_df[~pca_df['label']]["PC2"], c='red', alpha = 0.05)  # or c=pca_df["label"] if you have categories
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of image bounding boxes for images with a single bounding box ")
plt.grid(True)
plt.show()

#%%
# turn the indivisual observations into events with a start and end
ret = []
start = data.iloc[0]['timestamp']
first_image = ''
state = 'non_bird'
cnt = 0
for ind, row in data.iterrows():
    if state == 'non_bird':
        if row['bird']:
            state = 'bird'
            start = row['timestamp']
            first_image = row['filename']
            cnt = 0
    else:
        if not row['bird']:
            state = 'non_bird'
            end = row['timestamp']
            ret.append({'start': start, 'first_image':first_image, 'end':end, 'last_image': row['filename'], 'cnt_images': cnt, 'duration': (end-start).total_seconds()})
    cnt += 1

events = pd.DataFrame(ret)

#%%
# plot the events
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

for idx, row in events.iterrows():
    plt.hlines(y=idx, xmin=row['start'], xmax=row['end'], color='blue', linewidth=5)

plt.xlabel("Time")
plt.ylabel("Event ID")
plt.title("Timeline of Events")
#plt.yticks(events.index)  # show all event IDs on y-axis
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.xticks(rotation=45)  
plt.show()

#%%
# create video

import cv2
import os
import glob

# --- 1. Path to your images ---
image_folder = base_dir+"images/"  # folder with JPGs
image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
image_files = [item for item in image_files if item>='jonathan/images\\frame_00002312.jpg']
image_files = [item for item in image_files if item<='jonathan/images\\frame_00002513.jpg']

# --- 2. Read first image to get size ---
frame = cv2.imread(image_files[0])
height, width, layers = frame.shape
size = (width, height)

# Center crop to 480x480
crop_size = 480
x_start = (width - crop_size) // 2  # 80
y_start = 0

cropped = frame[y_start:y_start + crop_size,
              x_start:x_start + crop_size]

# Resize to 1080x1080
output = cv2.resize(cropped, (1080, 1080), interpolation=cv2.INTER_AREA)



# --- 3. Create VideoWriter object ---
out = cv2.VideoWriter(base_dir+'output/output_video2.mp4', 
                      cv2.VideoWriter_fourcc(*'mp4v'),  # codec
                      10,  # frames per second
                      (1080,1080))



# --- 4. Add each image to the video ---
for filename in image_files:
    img = cv2.imread(filename)
    cropped = img[y_start:y_start + crop_size,
                  x_start:x_start + crop_size]

    # Resize to 1080x1080
    output = cv2.resize(cropped, (1080, 1080), interpolation=cv2.INTER_AREA)

    out.write(output)  # add frame

# --- 5. Release everything ---
out.release()
print("Video saved as output_video.mp4")

