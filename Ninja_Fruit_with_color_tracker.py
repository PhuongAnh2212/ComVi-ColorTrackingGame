import cv2
import time
import random
import numpy as np

# Setup
cap = cv2.VideoCapture(0)

# HSV color ranges for green and purple
lower_purple = np.array([130, 30, 30])   
upper_purple = np.array([140, 255, 255])
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])

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

# Game variables
curr_Frame, prev_Frame, delta_time = 0, 0, 0
next_Time_to_Spawn, Fruit_Size, Spawn_Rate = 0, 30, 1
Speed, Score, Lives, Difficulty_level = [0, 5], 0, 15, 1
game_Over = False

# Load fruit images
fruit_images = [
    cv2.imread("assets/apple.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("assets/atiso.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("assets/berry.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("assets/grape.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("assets/idk.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("assets/lemon.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("assets/mango.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("assets/strawberry.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("assets/watermelon.png", cv2.IMREAD_UNCHANGED),
]

# Load background
background = cv2.imread('assets/background.jpg')

slash_purple = np.array([[]], np.int32)
slash_green = np.array([[]], np.int32)
slash_length = 19
Fruits = []

def overlay_image_alpha(img, img_overlay, pos):
    x, y = pos
    h, w = img_overlay.shape[:2]

    if y - h < 0 or x + w > img.shape[1] or y > img.shape[0]:
        return

    roi = img[y - h:y, x:x + w]
    img_overlay_rgb = img_overlay[..., :3]
    mask = img_overlay[..., 3:] / 255.0

    for c in range(3):
        roi[..., c] = roi[..., c] * (1 - mask[:, :, 0]) + img_overlay_rgb[..., c] * mask[:, :, 0]

    img[y - h:y, x:x + w] = roi

def Spawn_Fruits():
    fruit = {}
    random_x = random.randint(15, 600)
    fruit["Curr_position"] = [random_x, 440]
    fruit["Next_position"] = [0, 0]
    fruit["Image"] = random.choice(fruit_images)
    Fruits.append(fruit)

def Fruit_Movement(Fruits, speed):
    global Lives
    for fruit in Fruits[:]:
        if fruit["Curr_position"][1] < 20 or fruit["Curr_position"][0] > 650:
            Lives -= 1
            Fruits.remove(fruit)
            continue

        pos = fruit["Curr_position"]
        overlay_image_alpha(img, fruit["Image"], (pos[0], pos[1]))
        fruit["Next_position"][0] = fruit["Curr_position"][0] + speed[0]
        fruit["Next_position"][1] = fruit["Curr_position"][1] - speed[1]
        fruit["Curr_position"] = fruit["Next_position"]

# overtime slashing will improve the plotting result of slashing_fruit
def is_fruit_cut(fruit, slash_position):
    fruit_x, fruit_y = fruit["Curr_position"]
    h, w = fruit["Image"].shape[:2]
    slash_x, slash_y = slash_position
    return fruit_x <= slash_x <= fruit_x + w and fruit_y - h <= slash_y <= fruit_y

def process_color(mask, kalman_filter, slash, slash_Color, draw_bbox=False):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        kalman_filter.predict()
        return slash, slash_Color, None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 300:
        kalman_filter.predict()
        return slash, slash_Color, None

    if draw_bbox:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        cx = x + w_box // 2
        cy = y + h_box // 2
        cv2.rectangle(img, (x, y), (x + w_box, y + h_box), slash_Color, 2)
    else:
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            kalman_filter.predict()
            return slash, slash_Color, None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
    kalman_filter.correct(measurement)
    prediction = kalman_filter.predict()
    cx_kalman, cy_kalman = int(prediction[0]), int(prediction[1])

    cv2.circle(img, (cx_kalman, cy_kalman), 18, slash_Color, -1)
    slash = np.append(slash, [(cx_kalman, cy_kalman)], axis=0) if slash.size else np.array([(cx_kalman, cy_kalman)])
    while len(slash) >= slash_length:
        slash = np.delete(slash, 0, 0)

    return slash, slash_Color, (cx_kalman, cy_kalman)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue
    h, w, c = img.shape
    background_resized = cv2.resize(background, (w, h))
    img = cv2.flip(img, 1)

    # Perform color detection on original webcam frame
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Track purple and green
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_OPEN, kernel)
    mask_purple = cv2.morphologyEx(mask_purple, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    # Apply background for display
    #display_img = cv2.addWeighted(background_resized, 1, img, 1, 0)

    # Process colors using the masks
    slash_purple, color_purple, pos_purple = process_color(mask_purple, kalman_purple, slash_purple, (255, 0, 255), draw_bbox=True)
    slash_green, color_green, pos_green = process_color(mask_green, kalman_green, slash_green, (0, 255, 0), draw_bbox=True)

    # Update img to display_img for rendering fruits, slashes, and UI
    # add multi color tracking process, if 
    #img = display_img

    # Detect hits
    for pos, slash_color in [(pos_purple, color_purple), (pos_green, color_green)]:
        if pos is not None:
            for fruit in Fruits[:]:
                if is_fruit_cut(fruit, pos):
                    Score += 100
                    Fruits.remove(fruit)

    # Update difficulty
    if Score % 1000 == 0 and Score != 0:
        Difficulty_level = int(Score / 1000) + 1
        Spawn_Rate = Difficulty_level * 4 / 5
        Speed = [0, int(5 * Difficulty_level / 2)]

    # Draw polylines
    if slash_purple.size:
        cv2.polylines(img, [slash_purple.reshape((-1, 1, 2))], False, color_purple, 15, 0)
    if slash_green.size:
        cv2.polylines(img, [slash_green.reshape((-1, 1, 2))], False, color_green, 15, 0)

    # UI
    curr_Frame = time.time()
    delta_time = curr_Frame - prev_Frame
    FPS = int(1 / delta_time) if delta_time else 0
    cv2.putText(img, f"FPS: {FPS}", (int(w * 0.82), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 250, 0), 2)
    cv2.putText(img, f"Score: {Score}", (int(w * 0.35), 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
    cv2.putText(img, f"Level: {Difficulty_level}", (int(w * 0.01), 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 150), 5)
    cv2.putText(img, f"Lives: {Lives}", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    prev_Frame = curr_Frame

    if Lives <= 0:
        game_Over = True

    if not game_Over:
        if time.time() > next_Time_to_Spawn:
            Spawn_Fruits()
            next_Time_to_Spawn = time.time() + (1 / Spawn_Rate)
        Fruit_Movement(Fruits, Speed)
    else:
        cv2.putText(img, "GAME OVER", (int(w * 0.1), int(h * 0.6)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
        Fruits.clear()

    cv2.imshow("Purple Mask", mask_purple)
    cv2.imshow("Green Mask", mask_green)
    cv2.imshow("Fruit Ninja Color Tracker", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()