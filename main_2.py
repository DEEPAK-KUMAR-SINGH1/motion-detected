# motion_wave_detector.py
# Requirements: pip install opencv-python pyttsx3 numpy

import cv2
import numpy as np
import pyttsx3
import time

engine = pyttsx3.init()
engine.setProperty('rate', 160)

cap = cv2.VideoCapture(0)
time.sleep(1)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

wave_counter = 0
last_speech = 0
cooldown = 3  # seconds before repeating speech

print("👀 Starting motion detection... Press 'q' to quit.")

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    motion_positions = []

    for contour in contours:
        if cv2.contourArea(contour) < 1500:
            continue
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        motion_positions.append(x)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    text = "No Motion"

    if motion_detected:
        # Analyze horizontal movement to detect hand wave
        if len(motion_positions) >= 2:
            movement_range = max(motion_positions) - min(motion_positions)
            if movement_range > 150:
                wave_counter += 1
            else:
                wave_counter = max(0, wave_counter - 1)

        if wave_counter > 4:
            text = "Hand Waving Detected"
            now = time.time()
            if now - last_speech > cooldown:
                engine.say("Hand waving detected")
                engine.runAndWait()
                last_speech = now
            wave_counter = 0
        else:
            text = "Motion Detected"

    cv2.putText(frame1, text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Motion Detector", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
