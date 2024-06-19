import cv2
import mediapipe as mp
import pyautogui

import math


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Initialize video capture
cam = cv2.VideoCapture(0)

# Initialize MediaPipe Face Mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
draw = mp.solutions.drawing_utils

# Get screen width and height for pyautogui
screen_w, screen_h = pyautogui.size()

while True:
    ret, frame = cam.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the face mesh
    face_output = face_mesh.process(rgb_frame)
    face_landmark_points = face_output.multi_face_landmarks

    # Process the hand landmarks
    hand_output = hands.process(rgb_frame)
    hand_landmark_points = hand_output.multi_hand_landmarks

    frame_h, frame_w, _ = frame.shape

    # Process hand landmarks
    if hand_landmark_points:
        for hand in hand_landmark_points:
            draw.draw_landmarks(frame, hand)
            x8 = int(hand.landmark[8].x * frame_w)
            y8 = int(hand.landmark[8].y * frame_h)
            x4 = int(hand.landmark[4].x * frame_w)
            y4 = int(hand.landmark[4].y * frame_h)

            distance = calculate_distance(x8, y8, x4, y4)
            threshold_distance = 40  # Adjust this threshold

            if distance < threshold_distance:
                color = (0, 0, 255)  # Red color
            else:
                color1 = (0, 255, 0)  # Green color
                color2 = (255, 0, 0)  # Blue color

            cv2.circle(frame, (x8, y8), 15, color if distance < threshold_distance else color1, cv2.FILLED)
            cv2.circle(frame, (x4, y4), 15, color if distance < threshold_distance else color2, cv2.FILLED)

            # Click when the circles turn red
            if distance < threshold_distance:
                pyautogui.click()
                pyautogui.sleep(1)

    # Process face landmarks for eye control
    if face_landmark_points:
        landmarks = face_landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            if id == 1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        if (left[0].y - left[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)

    cv2.imshow('Eye Controlled Mouse', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()