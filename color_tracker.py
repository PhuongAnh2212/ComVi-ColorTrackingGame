import numpy as np
import cv2
import socket
import time

# Fonts
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 2

# Kalman filter setup
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

# HSV color ranges
lower_purple = np.array([130, 30, 30])   
upper_purple = np.array([140, 255, 255])

lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])

# Morph kernel for mask cleaning
kernel = np.ones((5,5), np.uint8)

# Capture
cap = cv2.VideoCapture(0)

prev_time = time.time()

def process_color(mask, color_name, box_color, kalman_filter, send_udp=False):
    global frame
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        kalman_filter.predict()
        return

    # Find largest contour
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    # Filter small contours (noise)
    if cv2.contourArea(cnt) < 300:
        kalman_filter.predict()
        return

    M = cv2.moments(cnt)
    if M['m00'] == 0:
        kalman_filter.predict()
        return

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    x, y, w, h = cv2.boundingRect(cnt)

    measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
    kalman_filter.correct(measurement)
    prediction = kalman_filter.predict()
    cx_kalman, cy_kalman = int(prediction[0]), int(prediction[1])

    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
    label = f"{color_name}: {cx_kalman},{cy_kalman}"
    cv2.putText(frame, label, (cx_kalman, cy_kalman), font, fontScale, box_color, thickness, cv2.LINE_AA)


# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for intuitive mirror view
    frame = cv2.flip(frame, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.blur(hsv, (3, 3))

    # Masks for each color
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological operations to clean masks
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, kernel)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    # Show individual masks (optional)
    cv2.imshow("Mask - Purple", mask_purple)
    cv2.imshow("Mask - Green", mask_green)

    # Process both colors
    process_color(mask_purple, "purple", (255, 0, 255), kalman_purple, send_udp=True)
    process_color(mask_green, "green", (0, 255, 0), kalman_green, send_udp=True)

    # Show FPS
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {fps}', (10, 30), font, 1, (0,255,0), 2)

    # Show final output
    cv2.imshow("Tracked Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()