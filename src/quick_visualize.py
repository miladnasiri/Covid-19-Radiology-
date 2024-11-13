import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from data.dataset import get_data_loaders
from models.model import CovidClassifier

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    print("\nEvaluating model on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    return all_preds, all_labels, accuracy

def plot_results(preds, labels, class_names, save_dir='outputs'):
    # Create output directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png')
    print(f"\nConfusion matrix saved to {save_dir/'confusion_matrix.png'}")
    plt.close()
    
    # Classification Report
    report = classification_report(labels, preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    with open(save_dir / 'classification_report.txt', 'w') as f:
        f.write(report)

def main():
    print("Starting visualization and evaluation...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    _, _, test_loader, class_to_idx = get_data_loaders(
        data_dir="data/raw/COVID-19_Radiography_Dataset",
        batch_size=32,
        num_workers=4
    )
    
    class_names = list(class_to_idx.keys())
    print(f"\nClasses: {class_names}")
    
    # Load model
    print("\nLoading model...")
    model = CovidClassifier(
        num_classes=len(class_names),
        model_name='efficientnet_b0'
    ).to(device)
    
    checkpoint = torch.load('outputs/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from checkpoint with validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Evaluate
    preds, labels, accuracy = evaluate_model(model, test_loader, device)
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    
    # Plot results
    print("\nGenerating visualizations...")
    plot_results(preds, labels, class_names)

if __name__ == "__main__":
    main()
