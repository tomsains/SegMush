import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from pycocotools.coco import COCO
from torchvision import transforms
from PIL import Image
import torch

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define transformations for the mask
mask_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.NEAREST),  # Avoid interpolation artifacts
    transforms.Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.uint8)).long())  # Preserve integer values
])


class CocoDataset(Dataset):
    def __init__(self, root, annotation, transform=None):
        self.root = root
        self.transform = transform
        with open(annotation) as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        
        # Create a mapping from image_id to annotations
        self.image_to_anns = {}
        for ann in self.annotations:
            if ann['image_id'] not in self.image_to_anns:
                self.image_to_anns[ann['image_id']] = []
            self.image_to_anns[ann['image_id']].append(ann)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root, img_info['file_name'])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        
        anns = self.image_to_anns.get(img_info['id'], [])
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        

        for ann in anns:
            # Assuming segmentation is a polygon
            seg = ann['segmentation']
            for s in seg:
                poly = np.array(s).reshape((len(s)//2, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], 1)
        
        # Apply transforms
        image = image_transform(image)
        mask = mask_transform(Image.fromarray(mask)).unsqueeze(0)  # Add channel dimension 
    
        return image, mask




class CocoDatasetSegmentation:
    def __init__(self, root, annotation, transform=None):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __getitem__(self, index):
        # Get image ID
        img_id = self.ids[index]
        
        # Load image
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = f"{self.root}/{img_info['file_name']}"
        img = Image.open(img_path).convert("RGB")
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Extract bounding boxes, labels, and masks
        boxes = []
        labels = []
        masks = []
        
        for ann in anns:
            # Bounding boxes
            x_min, y_min, width, height = ann['bbox']
            boxes.append([x_min, y_min, x_min + width, y_min + height])
            
            # Labels
            labels.append(ann['category_id'])
            
            # Masks
            mask = self.coco.annToMask(ann)
            masks.append(mask)
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Stack masks into a single NumPy array before converting to tensor
        masks = np.stack(masks, axis=0)  # Shape: [N, H, W]
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks
        }
        
        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

    def __len__(self):
        return len(self.ids)