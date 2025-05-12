import numpy as np
import cv2
import socket

# Network setup
UDP_IP_ADDRESS = "127.0.0.1"
UDP_PORT_NO = 22222
clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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
lower_purple = np.array([125, 40, 40])   # Lower hue, sat, and val
upper_purple = np.array([170, 255, 255]) # Higher hue

lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])

# Capture
cap = cv2.VideoCapture(0)

def process_color(mask, color_name, box_color, kalman_filter, send_udp=False):
    global frame
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        kalman_filter.predict()
        return

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
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

    if send_udp:
        # UDP message format
        if color_name == "purple":
            message = f"P:{-(cx_kalman - 320) * (3.7 / 320):.2f}"
        elif color_name == "green":
            message = f"G:{-(cx_kalman - 320) * (3.7 / 320):.2f}"
        else:
            message = f"{cx_kalman},{cy_kalman}"
        clientSock.sendto(bytes(message, 'utf-8'), (UDP_IP_ADDRESS, UDP_PORT_NO))

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.blur(hsv, (3, 3))

    # Masks for each color
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Show individual masks (optional)
    cv2.imshow("Mask - Purple", mask_purple)
    cv2.imshow("Mask - Green", mask_green)

    # Process both colors
    process_color(mask_purple, "purple", (255, 0, 255), kalman_purple, send_udp=True)
    process_color(mask_green, "green", (0, 255, 0), kalman_green, send_udp=True)

    # Show final output
    cv2.imshow("Tracked Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
