import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import ssd300_vgg16


class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        # Create a label mapping
        self.label_mapping = {label: idx for idx, label in enumerate(self.img_labels['class'].unique())}

        print(f"Columns in the CSV file: {self.img_labels.columns}")  # Debugging statement

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")

        # Get bounding boxes and labels
        boxes = []
        labels = []
        filename = self.img_labels.iloc[idx, 0]
        annotations = self.img_labels[self.img_labels['filename'] == filename]
        for _, row in annotations.iterrows():
            boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            labels.append(self.label_mapping[row['class']])  # Map string label to numerical value

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

# Update file path and image directory
annotations_file = r'C:\Users\cchen\PycharmProjects\7980-Capstone\ssd\test.v4i.tensorflow\train\_annotations.csv'
img_dir = r'C:\Users\cchen\PycharmProjects\7980-Capstone\ssd\test.v4i.tensorflow\train'

# Ensure the file and directory exist
if not os.path.isfile(annotations_file):
    raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
if not os.path.isdir(img_dir):
    raise NotADirectoryError(f"Image directory not found: {img_dir}")

# Main entry point
if __name__ == "__main__":
    # Create dataset and dataloader instances
    dataset = CustomDataset(annotations_file=annotations_file, img_dir=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model
    model = ssd300_vgg16(weights='SSD300_VGG16_Weights.DEFAULT')
    model.to(device)
    model.train()

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Number of epochs
    num_epochs = 10

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for images, targets in dataloader:
            optimizer.zero_grad()  # Clear gradients of all optimized tensors

            # Forward pass: compute predictions
            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backpropagate the loss
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

        lr_scheduler.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {losses.item():.4f}')
