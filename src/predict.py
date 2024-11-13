import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from models.model import CovidClassifier
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform():
    return A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def predict_image(model, image_path, transform, device, class_names):
    # Load and preprocess image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transforms
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    
    # Get top predictions
    top_prob, top_class = torch.topk(probabilities, len(class_names))
    
    return top_prob.cpu().numpy(), top_class.cpu().numpy()

def plot_prediction(image_path, probabilities, class_indices, class_names):
    # Plot image
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title('Input X-ray')
    plt.axis('off')
    
    # Plot probabilities
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(class_names))
    plt.barh(y_pos, probabilities * 100)
    plt.yticks(y_pos, [class_names[i] for i in class_indices])
    plt.xlabel('Probability (%)')
    plt.title('Prediction Probabilities')
    
    plt.tight_layout()
    save_path = Path('outputs') / f"{image_path.stem}_prediction.png"
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Predict COVID-19 from X-ray images')
    parser.add_argument('image_path', type=str, help='Path to the input X-ray image')
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    
    # Load model
    model = CovidClassifier(
        num_classes=len(class_names),
        model_name='efficientnet_b0'
    ).to(device)
    
    checkpoint = torch.load('outputs/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Make prediction
    image_path = Path(args.image_path)
    probabilities, class_indices = predict_image(
        model,
        image_path,
        get_transform(),
        device,
        class_names
    )
    
    # Plot and save results
    plot_prediction(image_path, probabilities, class_indices, class_names)
    
    # Print results
    print("\nPrediction Results:")
    print("-" * 50)
    for prob, class_idx in zip(probabilities, class_indices):
        print(f"{class_names[class_idx]}: {prob*100:.2f}%")

if __name__ == '__main__':
    main()
