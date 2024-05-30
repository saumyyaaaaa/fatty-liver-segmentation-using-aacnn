import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Custom Dataset class for loading liver images
class LiverDataset(Dataset):
    def __init__(self, normal_dir, abnormal_dir, transform=None):
        self.normal_dir = os.path.abspath(normal_dir).replace("\\", "/")
        self.abnormal_dir = os.path.abspath(abnormal_dir).replace("\\", "/")
        self.transform = transform
        self.normal_images = [os.path.join(self.normal_dir, img) for img in os.listdir(self.normal_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
        self.abnormal_images = [os.path.join(self.abnormal_dir, img) for img in os.listdir(self.abnormal_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
        self.labels = [0] * len(self.normal_images) + [1] * len(self.abnormal_images)
        self.images = self.normal_images + self.abnormal_images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except (PermissionError, FileNotFoundError) as e:
            print(f"Error opening image {img_path}: {e}")
            return None, None
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

# Load datasets
try:
    train_dataset = LiverDataset('D:/College/Semester 6/Research Internship/dataset/train_normal', 
                                 'D:/College/Semester 6/Research Internship/dataset/train_abnormal', 
                                 transform=transform)
    val_dataset = LiverDataset('D:/College/Semester 6/Research Internship/dataset/val_normal', 
                               'D:/College/Semester 6/Research Internship/dataset/val_abnormal', 
                               transform=transform)
    test_dataset = LiverDataset('D:/College/Semester 6/Research Internship/dataset/test_normal', 
                                'D:/College/Semester 6/Research Internship/dataset/test_abnormal', 
                                transform=transform)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure the dataset directories are correctly set up.")
    raise

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the Attention Block used in the network
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.conv1(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.conv2(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.conv3(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = out + x
        return out

# Define the main model architecture
class AttentionCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AttentionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.attention1 = AttentionBlock(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.attention2 = AttentionBlock(256)
        self.fc1 = nn.Linear(256*32*32, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.attention1(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.attention2(x)
        x = x.view(-1, 256*32*32)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Initialize the model
model = AttentionCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            if inputs is None or labels is None:
                continue
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] completed. {num_epochs-epoch-1} epochs left.")

        # Save the best model
        torch.save(model.state_dict(), 'best_model.pth')

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25)

# Testing function
def test_model(model, test_loader):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            if inputs is None or labels is None:
                continue
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total}%")

# Test the model
test_model(model, test_loader)
