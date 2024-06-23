import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        
        # Get bounding boxes and labels
        boxes = []
        labels = []
        annotations = self.img_labels[self.img_labels['image_id'] == self.img_labels.iloc[idx, 0]]
        for _, row in annotations.iterrows():
            boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            labels.append(row['class'])
        
        # Convert boxes and labels to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        if self.transform:
            image = self.transform(image)
        
        # Dictionary for image and its annotations
        target = {'boxes': boxes, 'labels': labels}
        
        return image, target

# Define transforms
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

# TODO: update file path and image directory
# Create dataset and dataloader instances
dataset = CustomDataset(annotations_file='/path/to/dataset/annotations.csv',
                        img_dir='/path/to/dataset/images',
                        transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# Define model
model = ssd300_vgg16(pretrained=True)
model.train()

# Optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = model.loss  # Note: You may need to define a custom loss function if your dataset structure is different

# Number of epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    for images, targets in dataloader:
        optimizer.zero_grad()          # Clear gradients of all optimized tensors
        outputs = model(images)        # Forward pass: compute predictions
        loss = criterion(outputs, targets)  # Compute the loss
        loss.backward()                # Backpropagate the loss
        optimizer.step()               # Update the model parameters
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
