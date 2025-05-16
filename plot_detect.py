import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from collections import defaultdict

# Fonts
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 2

def create_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    return kf

kalman_purple = create_kalman()
kalman_green = create_kalman()

lower_purple = np.array([130, 30, 30])   
upper_purple = np.array([140, 255, 255])
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])

kernel = np.ones((5, 5), np.uint8)

cap = cv2.VideoCapture(0)
prev_time = time.time()
start_time = time.time()
last_logged = 0

detection_log = defaultdict(lambda: {'green': 0, 'purple': 0, 'both': 0})

def process_color(mask, color_name, box_color, kalman_filter):
    global frame
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        kalman_filter.predict()
        return False
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    if cv2.contourArea(cnt) < 300:
        kalman_filter.predict()
        return False
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        kalman_filter.predict()
        return False
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    x, y, w, h = cv2.boundingRect(cnt)
    measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
    kalman_filter.correct(measurement)
    prediction = kalman_filter.predict()
    cx_kalman, cy_kalman = int(prediction[0]), int(prediction[1])
    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
    cv2.putText(frame, f"{color_name}: {cx_kalman},{cy_kalman}", (cx_kalman, cy_kalman),
                font, fontScale, box_color, thickness, cv2.LINE_AA)
    return True

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.blur(hsv, (3, 3))
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, kernel)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    green_detected = process_color(mask_green, "green", (0, 255, 0), kalman_green)
    purple_detected = process_color(mask_purple, "purple", (255, 0, 255), kalman_purple)

    elapsed = int(time.time() - start_time)
    if elapsed != last_logged:
        last_logged = elapsed
        detection_log[elapsed]['green'] = int(green_detected)
        detection_log[elapsed]['purple'] = int(purple_detected)
        detection_log[elapsed]['both'] = int(green_detected and purple_detected)

    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {fps}', (10, 30), font, 1, (0, 255, 0), 2)

    cv2.imshow("Tracked Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Auto-stop after 60 seconds
    if time.time() - start_time >= 60:
        print("Auto-stop after 60 seconds.")
        break

cap.release()
cv2.destroyAllWindows()

# Plot results
times = sorted(detection_log.keys())
#green_vals = [detection_log[t]['green'] for t in times]
purple_vals = [detection_log[t]['purple'] for t in times]
#both_vals = [detection_log[t]['both'] for t in times]

plt.figure(figsize=(10, 5))
#plt.plot(times, green_vals, label='Green Detected', color='green')
plt.plot(times, purple_vals, label='Purple Detected', color='purple')
#plt.plot(times, both_vals, label='Both Detected', color='black', marker='o', linestyle='-')
plt.xlabel('Time (seconds)')
plt.ylabel('Detection (1 = Yes, 0 = No)')
#plt.title('Color Detection of Both Colors Over Time (1 Minute)')
plt.legend()
plt.yticks([0, 1])
plt.grid(True)
plt.tight_layout()
plt.show()