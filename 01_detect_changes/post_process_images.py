# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 12:59:55 2025

@author: AndreasWombacher
"""

import os
import requests
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

from ultralytics import YOLO
import cv2


base_dir = "jonathan/"
# -------------------------
# Configuration
# -------------------------
WEBPAGE_URL = "http://192.168.178.150/"
DOWNLOAD_DIR = base_dir+"download/"
ALLOWED_EXTENSIONS = {".jpg"}

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")


# -------------------------
# Helper functions
# -------------------------
def is_allowed_file(url):
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def download_file(url):
    filename = os.path.basename(urlparse(url).path)
    local_path = os.path.join(DOWNLOAD_DIR, filename)

    if os.path.exists(local_path):
        return local_path

    print(f"Downloading: {url}")
    r = requests.get(url, stream=True, timeout=10)
    r.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    return local_path


def run_yolo_on_image(image_path):
    print(f"Running YOLO on image: {image_path}")
    results = model(image_path)

    for r in results:
        r.save(filename=image_path.replace(".", "_yolo."))

def get_image_from_url(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    # Convert bytes to numpy array
    image_array = np.frombuffer(response.content, dtype=np.uint8)
    
    # Decode image
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return img

def process_frame(frame, link):
    # Run inference
    results = model.predict(source=frame, conf=0.3)  # conf = confidence threshold
    
    flag = False
    # Draw bounding boxes
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
    
        for box, score, cls_id in zip(boxes, scores, class_ids):
            if cls_id == 14:  # COCO class 14 = bird
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Bird {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                flag = True
            # x1, y1, x2, y2 = map(int, box)
            # class_name = COCO_CLASSES[int(cls_id)]
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # cv2.putText(frame, f"{class_name} {score:.2f}", (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if flag:
        fname = link[link.rfind('/')+1:]
        cv2.imwrite(DOWNLOAD_DIR+fname, frame)


# Load a pretrained YOLOv8 model (trained on COCO dataset)
model = YOLO("yolov8x.pt")  # 'n' = nano (fast), other options: s, m, l, x

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 
    'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
    'toothbrush'
]

# -------------------------
# Main logic
# -------------------------
def main():
    response = requests.get(WEBPAGE_URL, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    links = set()
    for a in soup.find_all("a", href=True):
        full_url = urljoin(WEBPAGE_URL, a["href"])
        if is_allowed_file(full_url):
            links.add(full_url)

    print(f"Found {len(links)} file links")

    for link in links:
        print(link)
        try:
            # local_file = download_file(link)
            # ext = os.path.splitext(local_file)[1].lower()

            # if ext in {".jpg"}:
            #     run_yolo_on_image(local_file)
            frame = get_image_from_url(link)
            process_frame(frame, link)
        except Exception as e:
            print(f"Failed processing {link}: {e}")


if __name__ == "__main__":
    main()

link = "http://192.168.178.150/frame_1212.jpg"
link = "http://192.168.178.150/frame_12170.jpg"

link = "http://192.168.178.150/frame_12497.jpg"
link = "http://192.168.178.150/frame_12614.jpg"

link = "http://192.168.178.150/frame_1699.jpg"
frame = get_image_from_url(link)
process_frame(frame, link)
