# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 15:06:28 2025

@author: AndreasWombacher
"""

import cv2

base_dir = "jonathan/"
# Open webcam
cap = cv2.VideoCapture(0)


# Create background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

ii = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.flip(frame, 0)
    frame_orig = frame.copy()
    
    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Clean the mask (remove noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours of moving objects
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    flag = False
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:  # filter small objects
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            flag = True
        
    #cv2.imshow("Objects", frame)
    #cv2.imshow("Mask", fgmask)
    if flag:
        cv2.imwrite(f"/var/www/html/frame_{ii:08}.jpg", frame_orig)
        ii +=1
    #if cv2.waitKey(30) & 0xFF == ord('q'):
    #    break

#cap.release()
#cv2.destroyAllWindows()
