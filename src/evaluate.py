import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from data.dataset import get_data_loaders
from models.model import CovidClassifier

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()

def evaluate_model(model, test_loader, device, class_to_idx):
    model.eval()
    all_preds = []
    all_labels = []
    
    print("\nEvaluating model on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert indices to class names
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)
    
    # Print classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # Save report to file
    with open('outputs/classification_report.txt', 'w') as f:
        f.write(report)
    
    return cm, report

def predict_single_image(model, image_path, transform, device, class_to_idx):
    import cv2
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transforms
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)
        confidence, prediction = torch.max(prob, 1)
    
    # Convert prediction to class name
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    predicted_class = idx_to_class[prediction.item()]
    
    return predicted_class, confidence.item()

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    data_dir = "data/raw/COVID-19_Radiography_Dataset"
    _, _, test_loader, class_to_idx = get_data_loaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=4
    )
    
    # Load model
    model_path = 'outputs/best_model.pth'
    model = CovidClassifier(
        num_classes=len(class_to_idx),
        model_name='efficientnet_b0'
    ).to(device)
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation accuracy {checkpoint['val_acc']:.2f}%")
    
    # Evaluate model
    cm, report = evaluate_model(model, test_loader, device, class_to_idx)
    
    # Example of single image prediction
    sample_image = "data/raw/COVID-19_Radiography_Dataset/COVID/images/COVID-1.png"
    predicted_class, confidence = predict_single_image(
        model, 
        sample_image,
        test_loader.dataset.transform,
        device,
        class_to_idx
    )
    print(f"\nSample prediction for {sample_image}:")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence*100:.2f}%")

if __name__ == '__main__':
    main()
