import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from typing import Dict, Tuple, Optional
from pathlib import Path

class AdvancedCovidDataset(Dataset):
    """Advanced COVID-19 X-ray Dataset with sophisticated augmentations."""
    
    def __init__(
        self,
        data_frame,
        config: DataConfig,
        transform: Optional[A.Compose] = None,
        mode: str = "train"
    ):
        self.data = data_frame
        self.config = config
        self.mode = mode
        self.transform = transform or self._get_transforms()
        
    def _get_transforms(self) -> A.Compose:
        """Get albumentations transforms based on mode."""
        if self.mode == "train":
            return A.Compose([
                A.RandomResizedCrop(
                    height=self.config.image_size[0],
                    width=self.config.image_size[1],
                    scale=(0.8, 1.0)
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=30,
                    p=0.5
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50)),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=(3, 7)),
                ], p=0.5),
                A.OneOf([
                    A.OpticalDistortion(distort_limit=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.3),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
                ], p=0.3),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=self.config.image_size[0]//16,
                    max_width=self.config.image_size[1]//16,
                    min_holes=5,
                    fill_value=0,
                    p=0.3
                ),
                A.Normalize(),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(
                    height=self.config.image_size[0],
                    width=self.config.image_size[1]
                ),
                A.Normalize(),
                ToTensorV2(),
            ])
    
    def mixup(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup augmentation."""
        if random.random() > self.config.mixup_alpha:
            return image, label
            
        idx = random.randint(0, len(self.data) - 1)
        mixed_image = cv2.imread(self.data.iloc[idx]["filepath"])
        mixed_image = self.transform(image=mixed_image)["image"]
        mixed_label = torch.tensor(self.data.iloc[idx]["label"])
        
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
        image = lam * image + (1 - lam) * mixed_image
        label = lam * label + (1 - lam) * mixed_label
        
        return image, label
    
    def cutmix(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cutmix augmentation."""
        if random.random() > self.config.cutmix_alpha:
            return image, label
            
        idx = random.randint(0, len(self.data) - 1)
        mixed_image = cv2.imread(self.data.iloc[idx]["filepath"])
        mixed_image = self.transform(image=mixed_image)["image"]
        mixed_label = torch.tensor(self.data.iloc[idx]["label"])
        
        # Generate random box
        lam = np.random.beta(self.config.cutmix_alpha, self.config.cutmix_alpha)
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(image.size(), lam)
        
        # Apply cutmix
        image[:, bbx1:bbx2, bby1:bby2] = mixed_image[:, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
        label = lam * label + (1 - lam) * mixed_label
        
        return image, label
    
    def mosaic(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mosaic augmentation."""
        if random.random() > self.config.mosaic_prob:
            return image, label
            
        # Get three more random images
        indices = random.sample(range(len(self.data)), 3)
        images = [cv2.imread(self.data.iloc[i]["filepath"]) for i in indices]
        images = [self.transform(image=img)["image"] for img in images]
        labels = [torch.tensor(self.data.iloc[i]["label"]) for i in indices]
        
        # Create mosaic
        result_image = torch.zeros_like(image)
        result_label = torch.zeros_like(label)
        
        # Split image into 4 parts
        h, w = image.size(1), image.size(2)
        cx = int(random.uniform(w/4., 3*w/4.))
        cy = int(random.uniform(h/4., 3*h/4.))
        
        # Fill the quadrants
        result_image[:, :cy, :cx] = image[:, :cy, :cx]
        result_image[:, :cy, cx:] = images[0][:, :cy, -(w-cx):]
        result_image[:, cy:, :cx] = images[1][:, -(h-cy):, :cx]
        result_image[:, cy:, cx:] = images[2][:, -(h-cy):, -(w-cx):]
        
        # Weighted labels
        w1 = (cx * cy) / (w * h)
        w2 = ((w - cx) * cy) / (w * h)
        w3 = (cx * (h - cy)) / (w * h)
        w4 = ((w - cx) * (h - cy)) / (w * h)
        
        result_label = w1 * label + w2 * labels[0] + w3 * labels[1] + w4 * labels[2]
        
        return result_image, result_label
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        image = cv2.imread(row["filepath"])
        image = self.transform(image=image)["image"]
        label = torch.tensor(row["label"])
        
        if self.mode == "train":
            # Apply advanced augmentations
            image, label = self.mixup(image, label)
            image, label = self.cutmix(image, label)
            image, label = self.mosaic(image, label)
        
        return {
            "image": image,
            "label": label
        }
    
    def __len__(self) -> int:
        return len(self.data)
