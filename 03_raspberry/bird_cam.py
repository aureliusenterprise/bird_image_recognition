# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 23:51:36 2025

@author: AndreasWombacher
"""

import cv2
import csv
import numpy as np
import tflite_runtime.interpreter as tflite
import time
from datetime import datetime
import matplotlib.pyplot as plt


# ====== MODEL SETTINGS ======
MODEL_PATH = "bird_classifier_int8.tflite"

CROP_LEFT = 130
CROP_RIGHT = 185
CROP_BOTTOM = 150
TARGET = (160,160)  # width, height
THRESHOLD = 0.8

# ====== LOAD MODEL ======
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
print(interpreter.get_input_details())

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded")

# ====== OPEN CAMERA ======
cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# Try lower resolution to reduce CPU load
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("❌ Cannot open camera. Try another index or enable camera support.")

print("Camera opened")
ii = 0
with open("/var/www/html/data.csv","w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "bird", "timestamp", "filename", "probability","contours"])

    # ====== LOOP ======
    while True:
        ok, frame = cap.read()
        if not ok:
            print("⚠️ Failed to read frame")
            time.sleep(0.1)
            continue
        
        frame = cv2.flip(frame,1)
        frame = cv2.flip(frame,0)
        h, w = frame.shape[:2]
        frame_orig = frame.copy()

        fgmask = fgbg.apply(frame)
                # SAFETY CLAMP in case of wrong resolution
        # crop = frame_orig[
        #     0:max(0, h - CROP_BOTTOM),
        #     min(w, CROP_LEFT):max(0, w - CROP_RIGHT)
        # ]
    
        # # resize to model resolution
        # resized = cv2.resize(crop, TARGET)

        # uint8 input
        #inp = np.expand_dims(crop, axis=0).astype(np.uint8)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        flag = False
        arr = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                arr.append({'x':x, 'y':y, 'w':w, 'h':h})
                flag = True
        if flag:
            #frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB)
            # ================================
            # PREPROCESS: scale to [-1, 1] for MobileNetV2
            # ================================
            inp= frame_orig.astype(np.float32)
            inp = (inp / 127.5) - 1.0
            inp = np.expand_dims(inp, axis=0)
            # run model
            interpreter.set_tensor(input_details[0]['index'], inp)
            interpreter.invoke()

            output = interpreter.get_tensor(output_details[0]['index'])[0][0]
            prob = output / 255.0

            if prob < THRESHOLD:
                cv2.imwrite(f"/var/www/html/frame_{ii:08}.jpg", frame_orig)
                ii += 1
            else: 
                cv2.imwrite(f"/var/www/html/non/frame_{ii:08}.jpg", frame_orig)
                ii += 1
            writer.writerow([ii,flag, datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"), f"frame_{ii:8}.jpg",prob, str(arr) ]) 
 
cap.release()
