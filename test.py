import cv2
import numpy as np

# Color ranges in HSV
COLOR_RANGE = {
    'yellow': ([20, 100, 100], [40, 255, 255]),
    'red': ([0, 100, 100], [10, 255, 255]),
    # 'blue': ([90, 80, 70], [120, 255, 255]),
    # 'green': ([40, 80, 32], [70, 255, 255]),
}

# Color BGR for drawing
DRAW_COLORS = {
    'yellow': (0, 255, 255),
    'red': (0, 0, 255),
    # 'blue': (255, 0, 0),
    # 'green': (0, 255, 0),
}

# To store last known positions
last_positions = {color: None for color in COLOR_RANGE}

# Canvas for drawing paths
path_canvas = None

def detect_and_track(frame, color_name):
    global path_canvas

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower, upper = COLOR_RANGE[color_name]
    lower = np.array(lower)
    upper = np.array(upper)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (9, 9), 2)

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=2, minDist=frame.shape[0] // 4,
                               param1=100, param2=30, minRadius=10, maxRadius=200)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :1]:  # Only track the first detected circle
            x, y, r = i
            cv2.circle(frame, (x, y), r, DRAW_COLORS[color_name], 2)
            cv2.circle(frame, (x, y), 2, (255, 255, 255), 3)
            print(f"{color_name} ball found at ({x}, {y})")

            # Draw tracking line
            if last_positions[color_name] is not None:
                lx, ly = last_positions[color_name]
                cv2.line(path_canvas, (lx, ly), (x, y), DRAW_COLORS[color_name], 2)

            last_positions[color_name] = (x, y)

cap = cv2.VideoCapture(1)
ret, frame = cap.read()

if not ret:
    print("Failed to open camera.")
    cap.release()
    exit()

path_canvas = np.zeros_like(frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    for color in COLOR_RANGE:
        detect_and_track(frame, color)

    # Overlay the path canvas on the current frame
    output = cv2.addWeighted(frame, 0.8, path_canvas, 1, 0)

    cv2.imshow("Color Tracking", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
