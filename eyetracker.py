import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

screen_w, screen_h = pyautogui.size()
cam = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# ---- Settings ----
mouth_threshold = 0.02      # bigger = must open wider
click_delay = 1.2           # seconds between clicks
sensitivity = 10
smoothening = 5

last_click_time = 0
prev_x, prev_y = screen_w/2, screen_h/2

center_x, center_y = None, None

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb)

    frame_h, frame_w, _ = frame.shape

    if output.multi_face_landmarks:
        landmarks = output.multi_face_landmarks[0].landmark

        # -------- Cursor Movement (Nose) --------
        nose = landmarks[1]
        x = int(nose.x * frame_w)
        y = int(nose.y * frame_h)

        key = cv2.waitKey(1)
        if key == ord('c'):
            center_x, center_y = x, y
            print("CALIBRATED")

        if center_x is not None:
            dx = (x - center_x) * sensitivity
            dy = (y - center_y) * sensitivity

            screen_x = prev_x + dx
            screen_y = prev_y + dy

            screen_x = min(screen_w - 1, max(0, screen_x))
            screen_y = min(screen_h - 1, max(0, screen_y))

            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

        # -------- Mouth Open Detection --------
        top_lip = landmarks[13]
        bottom_lip = landmarks[14]

        mouth_distance = abs(top_lip.y - bottom_lip.y)

        if mouth_distance > mouth_threshold:
            if time.time() - last_click_time > click_delay:
                pyautogui.click()
                print("MOUTH CLICK")
                last_click_time = time.time()

    cv2.imshow("Eye Mouse", frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()