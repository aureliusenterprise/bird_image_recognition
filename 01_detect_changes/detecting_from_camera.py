# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 15:21:48 2025

@author: AndreasWombacher
"""

# pip install ultralytics opencv-python


from ultralytics import YOLO
import cv2

base_dir = "jonathan/"

# Load a pretrained YOLOv8 model (trained on COCO dataset)
model = YOLO("yolov8n.pt")  # 'n' = nano (fast), other options: s, m, l, x

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

# Load an image
# image_path = base_dir+f"output/frame10.jpg"
# image = cv2.imread(image_path)
cap = cv2.VideoCapture(0)

ii = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.flip(frame, 0)
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
            #x1, y1, x2, y2 = map(int, box)
            #class_name = COCO_CLASSES[int(cls_id)]
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(frame, f"{class_name} {score:.2f}", (x1, y1 - 10),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        

    #cv2.imshow("Objects", frame)
    if flag:
        cv2.imwrite(f"/var/www/html/frame_{ii:04}.jpg",frame)
        ii += 1
    #if cv2.waitKey(30) & 0xFF == ord('q'):
    #    break

#cap.release()
#cv2.destroyAllWindows()

