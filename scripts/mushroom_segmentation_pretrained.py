import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # For saving data to CSV
from data_loader import CocoDataset  # Ensure this is correctly implemented

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load dataset
train_dataset = CocoDataset(root='./data/train/', annotation='./data/train/_annotations.coco.json', transform=transforms.ToTensor())
val_dataset = CocoDataset(root='./data/valid/', annotation='./data/valid/_annotations.coco.json', transform=transforms.ToTensor())
test_dataset = CocoDataset(root='./data/test/', annotation='./data/test/_annotations.coco.json', transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
print(len(val_loader))

# Create a U-Net with a pre-trained ResNet34 backbone
model = smp.Unet(
    encoder_name="resnet34",        # Pre-trained backbone
    encoder_weights="imagenet",     # Use ImageNet pre-trained weights
    in_channels=3,                  # Input channels (e.g., RGB)
    classes=1,                      # Output classes (e.g., binary segmentation)
    activation=None,                # No activation for BCEWithLogitsLoss
).to(device)  # Move model to the device


# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss with logits
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to calculate IoU
def calculate_iou(preds, targets):
    preds = torch.sigmoid(preds) > 0.5  # Apply sigmoid before thresholding
    targets = targets > 0.5
    intersection = (preds & targets).float().sum((1, 2))
    union = (preds | targets).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

# Training loop
num_epochs = 50

# Initialize lists to store metrics
epoch_data = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_iou = 0
    
    # Training phase
    for images, masks in train_loader:
        # Move data to the device
        images = images.to(device)
        masks = masks.to(device).float()  # Ensure masks are float
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate IoU
        iou = calculate_iou(outputs, masks)
        
        train_loss += loss.item()
        train_iou += iou.item()
    
    # Calculate average training loss and IoU
    avg_train_loss = train_loss / len(train_loader)
    avg_train_iou = train_iou / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_iou = 0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.float()
            
            outputs = model(images)
            #outputs = torch.sigmoid(outputs)  # Apply sigmoid
            
            loss = criterion(outputs, masks)
            iou = calculate_iou(outputs, masks)
            
            val_loss += loss.item()
            val_iou += iou.item()
    
    # Calculate average validation loss and IoU
    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)
    
    # Print metrics for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")
    
    # Store metrics for this epoch
    epoch_data.append({
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'train_iou': avg_train_iou,
        'val_loss': avg_val_loss,
        'val_iou': avg_val_iou
    })

# Save metrics to a CSV file
df = pd.DataFrame(epoch_data)
df.to_csv('training_metrics.csv', index=False)
print("Metrics saved to training_metrics.csv")

# Visualization (optional)
with torch.no_grad():
    fig, ax = plt.subplots(6, 2)

  
    for images, masks in val_loader:
        images, masks = images.to(device), masks.to(device)
        masks = masks.float()
        
        outputs = model(images)
        #outputs = torch.sigmoid(outputs)  # Apply sigmoid

        for i in range(images.shape[0]):
            img = images[i].squeeze(1).cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            
            ax[i,0].imshow(img, cmap="gray")
            ax[i,0].imshow(outputs[i, 0].squeeze(1).cpu().numpy(), alpha=0.5, cmap="PiYG", vmin=0, vmax=1)
            ax[i,1].imshow(img, cmap="gray")
            ax[i,1].imshow(masks[i, 0].squeeze(1).cpu().numpy(), alpha=0.5, cmap="PiYG", vmin=0, vmax=1)
    #plt.savefig("mushroom images")
    plt.show()


import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

# Create a custom colormap: 0 is transparent, 1 is green
colors = [(0, 0, 0, 0), (0, 1, 0, .5)]  # RGBA: (R, G, B, A)
cmap = ListedColormap(colors)

with torch.no_grad():
    fig, ax = plt.subplots(3, 3)
    img_list = []
    output_list = []
    mask_list = []
  
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        masks = masks.float()
        
        outputs = model(images)
        #outputs = torch.sigmoid(outputs)  # Apply sigmoid

        for i in range(images.shape[0]):
            img = images[i].squeeze(1).cpu().numpy().transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            img_list.append(img)
            output_list.append(outputs[i, 0].squeeze(1).cpu().numpy())
            mask_list.append(masks[i, 0].squeeze(1).cpu().numpy())
            print(calculate_iou(outputs[i], masks[i]))
            ax[i,0].imshow(img, cmap="gray")
            ax[i,0].imshow(outputs[i, 0].squeeze(1).cpu().numpy(), alpha=0.5, cmap=cmap, vmin=0, vmax=1)
            ax[i,1].imshow(img, cmap="gray")
            ax[i,1].imshow(masks[i, 0].squeeze(1).cpu().numpy(), alpha=0.5, cmap=cmap, vmin=0, vmax=1)
            ax[i,2].imshow(masks[i, 0].squeeze(1).cpu().numpy(), alpha=0.5, cmap="grey", vmin=0, vmax=1)
            ax[i,2].imshow(outputs[i, 0].squeeze(1).cpu().numpy(), alpha=0.5, cmap=cmap, vmin=0, vmax=1)
            
       
    plt.show()

    np.save(arr={"images": img_list, "masks": mask_list, "predictions": output_list}, file="test_images_predictions", allow_pickle=True)