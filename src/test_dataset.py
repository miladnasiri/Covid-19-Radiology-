import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

class CovidXrayDataset:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def analyze_dataset(self):
        print("\nDataset Statistics:")
        print("-" * 50)
        
        total_images = 0
        class_counts = {}
        
        for class_name in self.classes:
            image_dir = os.path.join(self.base_dir, class_name, 'images')
            if not os.path.exists(image_dir):
                print(f"Warning: Directory not found - {image_dir}")
                continue
                
            n_images = len([f for f in os.listdir(image_dir) if f.endswith('.png')])
            class_counts[class_name] = n_images
            total_images += n_images
            print(f"{class_name}: {n_images} images")
        
        print("-" * 50)
        print(f"Total images: {total_images}")
        
        return class_counts

    def verify_image_loading(self):
        print("\nVerifying image loading...")
        corrupted_images = []
        
        for class_name in self.classes:
            image_dir = os.path.join(self.base_dir, class_name, 'images')
            print(f"\nChecking {class_name} images...")
            
            total_images = len([f for f in os.listdir(image_dir) if f.endswith('.png')])
            for i, img_name in enumerate(os.listdir(image_dir)):
                if not img_name.endswith('.png'):
                    continue
                    
                img_path = os.path.join(image_dir, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        raise Exception("Image loaded as None")
                    if i % 100 == 0:  # Show progress every 100 images
                        print(f"Processed {i}/{total_images} images")
                except Exception as e:
                    corrupted_images.append((img_path, str(e)))
        
        if corrupted_images:
            print("\nCorrupted images found:")
            for path, error in corrupted_images:
                print(f"- {path}: {error}")
        else:
            print("\nAll images loaded successfully!")

def check_gpu():
    print("\nChecking GPU availability:")
    print("-" * 50)
    if torch.cuda.is_available():
        print(f"CUDA is available!")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Will use CPU.")
    print("-" * 50)

def main():
    # Check GPU
    check_gpu()
    
    # Test dataset
    dataset_path = "data/raw/COVID-19_Radiography_Dataset"
    print(f"\nTesting dataset at: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
    
    dataset = CovidXrayDataset(dataset_path)
    
    # Analyze dataset
    class_counts = dataset.analyze_dataset()
    
    # Verify image loading
    dataset.verify_image_loading()

if __name__ == "__main__":
    main()
