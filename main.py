import cv2
import pyautogui
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

class EyeTrackingCNN(nn.Module):
    def __init__(self):
        super(EyeTrackingCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 10, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 10)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the trained model
model = EyeTrackingCNN()
model.load_state_dict(torch.load('eye_tracking_cnn.pth'))
model.eval()

def preprocess_image(image):
    image = cv2.resize(image, (50, 100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    return torch.Tensor(image)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Moving average parameters
window_size = 5
screen_x_buffer = deque(maxlen=window_size)
screen_y_buffer = deque(maxlen=window_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_region = roi_color[ey:ey+eh, ex:ex+ew]
            input_tensor = preprocess_image(eye_region)
            with torch.no_grad():
                output = model(input_tensor)
                screen_x = int(output[0][0].item() * pyautogui.size().width)
                screen_y = int(output[0][1].item() * pyautogui.size().height)
                
                # Append to buffer
                screen_x_buffer.append(screen_x)
                screen_y_buffer.append(screen_y)
                
                # Calculate moving average
                avg_screen_x = int(np.mean(screen_x_buffer))
                avg_screen_y = int(np.mean(screen_y_buffer))
                
                pyautogui.moveTo(avg_screen_x, avg_screen_y)

    cv2.imshow('Eye Controlled Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
