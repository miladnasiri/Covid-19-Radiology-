import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from data.dataset import get_data_loaders
from models.model import CovidClassifier

def create_sample_predictions(model, test_loader, device, num_samples=3):
    """Create visualizations for random samples from each class"""
    # Create output directory
    output_dir = Path('predictions')
    output_dir.mkdir(exist_ok=True)
    
    # Get class names
    class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    #class_to_idx = test_loader.dataset.class_to_idx
    
    # Dictionary to store samples per class
    class_samples = {class_name: [] for class_name in class_names}
    
    # Collect samples
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image']
            labels = batch['label']
            
            # Get predictions
            outputs = model(images.to(device))
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            # Store samples
            for img, label, pred, prob in zip(images, labels, preds, probs):
                true_class = class_names[label]
                if len(class_samples[true_class]) < num_samples:
                    class_samples[true_class].append({
                        'image': img.cpu().numpy(),
                        'true_label': true_class,
                        'pred_label': class_names[pred],
                        'probabilities': prob.cpu().numpy()
                    })
            
            # Check if we have enough samples
            if all(len(samples) >= num_samples for samples in class_samples.values()):
                break
    
    # Create visualizations
    for class_name, samples in class_samples.items():
        for i, sample in enumerate(samples):
            plt.figure(figsize=(15, 5))
            
            # Plot image
            plt.subplot(1, 2, 1)
            img = sample['image'].transpose(1, 2, 0)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            plt.imshow(img)
            plt.title(f'True: {sample["true_label"]}\nPredicted: {sample["pred_label"]}')
            plt.axis('off')
            
            # Plot probabilities
            plt.subplot(1, 2, 2)
            bars = plt.bar(class_names, sample['probabilities'] * 100)
            plt.title('Class Probabilities')
            plt.xlabel('Class')
            plt.ylabel('Probability (%)')
            plt.xticks(rotation=45)
            
            # Add probability values on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save figure
            save_path = output_dir / f"{class_name}_sample_{i+1}.png"
            plt.savefig(save_path)
            plt.close()
            print(f"Saved prediction visualization for {class_name} (Sample {i+1})")
    
    return output_dir

def main():
    print("Starting sample visualization process...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    _, _, test_loader, class_to_idx = get_data_loaders(
        data_dir="data/raw/COVID-19_Radiography_Dataset",
        batch_size=32,
        num_workers=4
    )
    
    # Load model
    print("Loading model...")
    model = CovidClassifier(
        num_classes=len(class_to_idx),
        model_name='efficientnet_b0'
    ).to(device)
    
    checkpoint = torch.load('outputs/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from checkpoint with validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Create visualizations
    print("\nGenerating sample predictions...")
    output_dir = create_sample_predictions(model, test_loader, device)
    
    # Print summary
    print(f"\nAll visualizations have been saved to {output_dir}")
    print("\nResults added to predictions directory. Use these images in your README or documentation.")
    
    # Generate markdown for README
    print("\nCopy this section to your README.md:")
    print("\n## Sample Predictions")
    print("\n### COVID-19 Samples")
    print("![COVID Sample 1](predictions/COVID_sample_1.png)")
    print("![COVID Sample 2](predictions/COVID_sample_2.png)")
    print("\n### Lung Opacity Samples")
    print("![Lung Opacity Sample 1](predictions/Lung_Opacity_sample_1.png)")
    print("![Lung Opacity Sample 2](predictions/Lung_Opacity_sample_2.png)")
    print("\n### Normal Samples")
    print("![Normal Sample 1](predictions/Normal_sample_1.png)")
    print("![Normal Sample 2](predictions/Normal_sample_2.png)")
    print("\n### Viral Pneumonia Samples")
    print("![Viral Pneumonia Sample 1](predictions/Viral_Pneumonia_sample_1.png)")
    print("![Viral Pneumonia Sample 2](predictions/Viral_Pneumonia_sample_2.png)")

if __name__ == "__main__":
    main()
