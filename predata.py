import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# Load the data
with open('eye_tracking_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# Ensure data is in the correct shape
X_train = np.array(X_train).reshape(-1, 50, 100)
X_test = np.array(X_test).reshape(-1, 50, 100)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Define transformations for data augmentation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Custom dataset class with augmentation
class EyeDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.Tensor(image).unsqueeze(0)  # Convert to 1xHxW tensor
        return image, torch.tensor(label, dtype=torch.float32)  # Ensure labels are float tensors

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Create DataLoader with augmentation
train_dataset = EyeDataset(X_train, y_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = EyeDataset(X_val, y_val, transform=None)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class EyeTrackingCNN(nn.Module):
    def __init__(self):
        super(EyeTrackingCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 10, 256)  # Adjust size based on output
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 10)  # Adjust size based on output
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer with weight decay
model = EyeTrackingCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Training with early stopping
epochs = 50
patience = 5
best_loss = float('inf')
trigger_times = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Epoch {epoch+1}/{epochs}, Training Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss}')

    if val_loss < best_loss:
        best_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), 'eye_tracking_cnn.pth')  # Save the best model
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping!")
            break
