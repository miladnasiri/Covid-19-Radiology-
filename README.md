# COVID-19 X-Ray Classification Project

## Results Summary
- **Training Accuracy:** 97.38%
- **Validation Accuracy:** 95.89%
- **Test Accuracy:** 96.46%

## Model Performance Metrics
```
                 precision    recall  f1-score   support

          COVID       0.99      0.98      0.99       362
   Lung_Opacity      0.96      0.94      0.95       602
         Normal      0.95      0.98      0.97      1019
Viral Pneumonia      0.98      0.94      0.96       134

       accuracy                           0.96      2117
      macro avg      0.97      0.96      0.97      2117
   weighted avg      0.96      0.96      0.96      2117
```

## Training Progress
![Training Progress](https://raw.githubusercontent.com/miladnasiri/Covid-19-Radiology-/main/wandb/run-20241113_164637-16vcktjk/files/media/images/train_acc_30.png)

## Architecture and Implementation
- Model: EfficientNet-B0
- Framework: PyTorch
- Training Features:
  - Mixed Precision Training
  - Learning Rate Scheduling
  - Advanced Data Augmentation
  - Weighted Loss Functions

## Dataset Information
- Total Images: 21,165
- Classes:
  - COVID: 3,616 images
  - Lung Opacity: 6,012 images
  - Normal: 10,192 images
  - Viral Pneumonia: 1,345 images

## Training Curves
![Learning Rate](https://raw.githubusercontent.com/miladnasiri/Covid-19-Radiology-/main/wandb/run-20241113_164637-16vcktjk/files/media/images/learning_rate_30.png)

## Validation Metrics
![Validation Accuracy](https://raw.githubusercontent.com/miladnasiri/Covid-19-Radiology-/main/wandb/run-20241113_164637-16vcktjk/files/media/images/val_acc_30.png)

## Project Links
- [Weights & Biases Dashboard](https://wandb.ai/miladnassiri92-topnetwork/covid-xray-classification/runs/16vcktjk)
- [GitHub Repository](https://github.com/miladnasiri/Covid-19-Radiology-)

## Setup and Usage
```bash
# Clone repository
git clone https://github.com/miladnasiri/Covid-19-Radiology-.git
cd Covid-19-Radiology-

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Train model
python src/train.py

# Evaluate model
python src/evaluate.py
```

## Model Architecture
- Base Model: EfficientNet-B0
- Custom Head:
  - Global Average Pooling
  - Dropout (0.3)
  - Dense Layer (512 units)
  - ReLU Activation
  - Dropout (0.3)
  - Output Layer (4 classes)

## Training Configuration
- Batch Size: 32
- Initial Learning Rate: 0.001
- Optimizer: AdamW
- Learning Rate Schedule: Cosine Annealing
- Data Augmentation:
  - Random Resized Crop
  - Horizontal Flip
  - Random Brightness/Contrast
  - Random Rotation
  - Gaussian Noise/Blur
