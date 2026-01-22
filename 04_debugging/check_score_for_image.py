# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 11:08:52 2026

@author: AndreasWombacher
"""

import cv2
import csv
import numpy as np
#import tflite_runtime.interpreter as tflite
import tensorflow as tf
import time
from datetime import datetime
import requests
import matplotlib.pyplot as plt

# ================================
# SETTINGS
# ================================
base_dir = ''

MODEL_PATH = "bird_classifier_int8.tflite"

CROP_LEFT = 115
CROP_RIGHT = 205
CROP_BOTTOM = 110
CROP_TOP = 50
TARGET_SIZE = (160,160)  # width, height
THRESHOLD = 0.1

ORIG_W = 640
ORIG_H = 480

# CROPPED_W = ORIG_W - CROP_LEFT - CROP_RIGHT
# CROPPED_H = ORIG_H - CROP_BOTTOM

# ====== LOAD MODEL ======
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
print(interpreter.get_input_details())

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded")
print(input_details)

# URL of the image
url = "http://192.168.178.150/frame_00009131.jpg"
url = "http://192.168.178.150/non/frame_00006155.jpg"
url = "http://192.168.178.150/frame_00007213.jpg"
url = "http://192.168.178.150/frame_00007293.jpg"
url = "http://192.168.178.150/frame_00007437.jpg"
url = "http://192.168.178.150/non/frame_00000013.jpg"
url = "http://192.168.178.150/non/frame_00003524.jpg"

resp = requests.get(url)
img_arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
frame_orig = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)   # BGR uint8
#%%
# load image from file 
#frame_orig = cv2.imread(base_dir+"images/non/frame_00003524.jpg")
frame_orig = cv2.imread(base_dir+"images/frame_00000130.jpg")
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

frame = preprocess(frame_orig)

frame_disp2 = (frame + 1.0) / 2.0
frame_disp2 = frame_disp2*255
frame_disp2 = frame_disp2.astype(int)
plt.imshow(frame_disp2)
plt.axis('off')
plt.show()

inp = np.expand_dims(frame, axis=0).astype(np.float32)
# run model
interpreter.set_tensor(input_details[0]['index'], inp)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])[0][0]
print(output)

