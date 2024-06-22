import cv2
import pyautogui
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

import cv2
import pyautogui
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

def capture_eye_images(num_images=2000):
    # Load Haar cascades for face and eye detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)
    images = []
    coordinates = []

    screen_width, screen_height = pyautogui.size()

    for i in range(num_images):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                eye_region = roi_color[ey:ey+eh, ex:ex+ew]
                eye_region = cv2.resize(eye_region, (50, 100))
                eye_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                images.append(eye_region)

                screen_x, screen_y = pyautogui.position()
                coordinates.append((screen_x / screen_width, screen_y / screen_height))

                # Display the eye region
                cv2.imshow('Eye Region', eye_region)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return images, coordinates

    cap.release()
    cv2.destroyAllWindows()

    return images, coordinates


    cap.release()
    cv2.destroyAllWindows()

    return images, coordinates

def preprocess_images(images):
    processed_images = []
    for img in images:
        img = cv2.resize(img, (50, 100))
        img = img / 255.0  # Normalize pixel values
        processed_images.append(img)
    return np.array(processed_images)

# Capture and preprocess the images
eye_images, eye_coordinates = capture_eye_images()
processed_eye_images = preprocess_images(eye_images)
X_train, X_test, y_train, y_test = train_test_split(processed_eye_images, eye_coordinates, test_size=0.1, random_state=42)

# Save the data to a file
with open('eye_tracking_data.pkl', 'wb') as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)
