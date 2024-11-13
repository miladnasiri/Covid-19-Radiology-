import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

class CovidXrayDataset(Dataset):
    def __init__(self, data_dir, filenames, labels, transform=None, phase='train'):
        self.data_dir = data_dir
        self.filenames = filenames
        self.labels = labels
        self.phase = phase
        self.transform = transform or self._get_transforms()
        
    def _get_transforms(self):
        if self.phase == 'train':
            return A.Compose([
                A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50)),
                    A.GaussianBlur(blur_limit=(3, 7)),
                ], p=0.3),
                A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(height=224, width=224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    # Get all image paths and labels
    image_paths = []
    labels = []
    class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name, 'images')
        for img_name in os.listdir(class_dir):
            if img_name.endswith('.png'):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_to_idx[class_name])
    
    # Split data
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    # Create datasets
    train_dataset = CovidXrayDataset(data_dir, train_paths, train_labels, phase='train')
    val_dataset = CovidXrayDataset(data_dir, val_paths, val_labels, phase='val')
    test_dataset = CovidXrayDataset(data_dir, test_paths, test_labels, phase='test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_to_idx
