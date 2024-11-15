
# 🔬 COVID-19 X-Ray Classification using EfficientNet with Attention Mechanism
This project implements an advanced deep learning system for COVID-19 detection from chest X-rays using the EfficientNet architecture with custom modifications. The model's core strength lies in its attention mechanism (CBAM - Convolutional Block Attention Module), which helps it focus on critical areas of X-ray images, similar to how radiologists examine specific regions for diagnosis. The system processes X-ray images through several key stages: first, the image is normalized and resized to 224×224 pixels; then, it passes through the EfficientNet backbone, which extracts important features from the image. The attention mechanism then highlights relevant areas, particularly those showing potential COVID-19 indicators. Finally, through a series of dense layers with dropout for regularization, the model classifies the image into one of four categories: COVID-19, Lung Opacity, Normal, or Viral Pneumonia.
What makes this implementation unique is its combination of high accuracy (96.46% overall, with 99% precision for COVID-19 cases) and practical efficiency (0.3 seconds per image processing time). The model achieves this through several technical innovations: mixed precision training for faster processing, cosine learning rate scheduling for better convergence, and a custom-weighted loss function to handle class imbalance in the dataset. The training data includes over 21,000 X-ray images, ensuring robust performance across different image qualities and conditions.
In real-world applications, this system serves as a valuable tool for medical professionals, offering rapid preliminary screening and a reliable second opinion for radiologists. The model's small size (23MB) and fast inference time make it practical for deployment in various medical settings, from large hospitals to smaller clinics with limited computational resources.

## 📊 Quick Overview
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Accuracy](https://img.shields.io/badge/Accuracy-96.46%25-success)
![License](https://img.shields.io/badge/License-MIT-green)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset%20Award%20Winner-blue)

## 🎯 Project Highlights
- **High Accuracy**: 96.46% on test set
- **Real-world Application**: Medical diagnosis support
- **Advanced Architecture**: EfficientNet with custom modifications
- **Production-Ready**: Implements best practices and modern techniques

## 📊 Dataset Details
### COVID-19 Radiography Database
This project utilizes the award-winning COVID-19 Radiography Database, created through international collaboration between:
- Qatar University, Doha, Qatar
- University of Dhaka, Bangladesh
- Medical professionals from Pakistan and Malaysia

### Dataset Evolution & Characteristics
```
Initial Release:             Current Version:
- COVID-19: 219 images      - COVID-19: 3,616 images
- Normal: 1,341 images      - Normal: 10,192 images
- Viral Pneumonia: 1,345    - Lung Opacity: 6,012 images
                           - Viral Pneumonia: 1,345 images
Total: 21,165 images
```

#### Technical Specifications
- **Format**: PNG format
- **Resolution**: 299×299 pixels
- **Type**: Grayscale chest X-rays
- **Annotations**: Includes lung masks
- **Quality**: Medical-grade, verified

## 🌟 Model Performance
| Class            | Precision | Recall | F1-Score |
|-----------------|-----------|---------|----------|
| COVID           | 0.99      | 0.98    | 0.99     |
| Lung Opacity    | 0.96      | 0.94    | 0.95     |
| Normal          | 0.95      | 0.98    | 0.97     |
| Viral Pneumonia | 0.98      | 0.94    | 0.96     |

## 🔮 Sample Predictions & Visualizations

### Training Progress
![Training Progress](https://raw.githubusercontent.com/miladnasiri/Covid-19-Radiology-/main/wandb/run-20241113_164637-16vcktjk/files/media/plots/train_acc_30_d5c46d60.png)

### Confusion Matrix
![Confusion Matrix](https://raw.githubusercontent.com/miladnasiri/Covid-19-Radiology-/main/outputs/confusion_matrix.png)

### Sample Predictions by Class

#### COVID-19 Cases
![COVID Sample](https://raw.githubusercontent.com/miladnasiri/Covid-19-Radiology-/main/predictions/COVID_sample_1.png)
- Confidence: 99.2%
- Clear identification of COVID-19 patterns

#### Lung Opacity Cases
![Lung Opacity Sample](https://raw.githubusercontent.com/miladnasiri/Covid-19-Radiology-/main/predictions/Lung_Opacity_sample_1.png)
- Confidence: 96.5%
- Distinct opacity patterns detected

#### Normal Cases
![Normal Sample](https://raw.githubusercontent.com/miladnasiri/Covid-19-Radiology-/main/predictions/Normal_sample_1.png)
- Confidence: 98.1%
- Clear healthy lung patterns

#### Viral Pneumonia Cases
![Viral Pneumonia Sample](https://raw.githubusercontent.com/miladnasiri/Covid-19-Radiology-/main/predictions/Viral_Pneumonia_sample_1.png)
- Confidence: 97.3%
- Distinguished from COVID-19 patterns

### Key Visual Findings
- Clear differentiation between COVID-19 and other conditions
- High confidence predictions across all classes
- Consistent performance on various image qualities
- Robust to different X-ray capture conditions

## 🏗️ Model Architecture
```mermaid
graph TD
    A[Input Layer 224x224x3] --> B[EfficientNet-B0]
    B --> C[Feature Maps]
    B --> D[Skip Connections]
    B --> E[Attention]
    C --> F[Global Average Pooling]
    D --> F
    E --> F
    F --> G[Dropout 0.5]
    G --> H[Dense 512 + ReLU]
    H --> I[Dropout 0.3]
    I --> J[Output Layer]
    J --> K[Softmax]
```

## 💡 Technical Implementation

### Data Preprocessing
```python
def preprocess_image(image):
    # Resize to model input size
    image = cv2.resize(image, (224, 224))
    
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Normalize pixel values
    image = image / 255.0
    
    # Apply standardization
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    
    return image
```

### Advanced Augmentation Pipeline
```python
augmentation = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.Transpose(p=0.5),
    A.OneOf([
        A.IAAAdditiveGaussianNoise(),
        A.GaussNoise(),
    ], p=0.2),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.1),
        A.IAAPiecewiseAffine(p=0.3),
    ], p=0.2),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.IAASharpen(),
        A.IAAEmboss(),
        A.RandomBrightnessContrast(),
    ], p=0.3),
    A.HueSaturationValue(p=0.3),
])
```

### Training Features
- Mixed Precision Training (FP16)
- Gradient Clipping & Accumulation
- Cosine Learning Rate Scheduling
- Early Stopping with Patience
- AdamW with Weight Decay
- Label Smoothing
- Class Weight Balancing

## 📈 Results
```
Final Metrics:
- Training Accuracy: 97.38%
- Validation Accuracy: 95.89%
- Test Accuracy: 96.46%
- Training Loss: 0.0782
- Validation Loss: 0.1432
```

## 🔧 Installation & Usage
```bash
# Clone repository
git clone https://github.com/miladnasiri/Covid-19-Radiology-.git

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model
python src/train.py

# Evaluate
python src/evaluate.py
```

## 🚀 Model Deployment
The trained model can be deployed using Docker and Flask, providing both a REST API and a web interface for real-time predictions.
![WebInterface](https://github.com/miladnasiri/Covid-19-Radiology-Detection/blob/38913ed9d34c1b09c4275cd53cbbfda11b8e7a0e/webInterface.png)
### Docker Deployment
```bash
# Build Docker image
docker build -t covid-xray-classifier .

# Run container
docker run -p 5000:5000 covid-xray-classifier
```

### Web Interface & API Endpoints
- **Web Interface**: Access `http://localhost:5000` for an interactive UI
- **REST API**: 
  - `POST /predict`: Send X-ray images for classification
  - `GET /health`: Check service health

### API Usage Example
```python
import requests

# Predict using an X-ray image
files = {'file': open('path/to/xray.png', 'rb')}
response = requests.post('http://localhost:5000/predict', files=files)
prediction = response.json()

# Example response:
# {
#     "class": "COVID",
#     "confidence": 0.992,
#     "probabilities": {
#         "COVID": 0.992,
#         "Lung_Opacity": 0.005,
#         "Normal": 0.002,
#         "Viral Pneumonia": 0.001
#     }
# }
```

### Deployment Files Structure
```
covid19_xray_classification/
├── Dockerfile              # Docker configuration
├── requirements_deploy.txt # Deployment dependencies
├── app.py                 # Flask application
└── src/
    └── models/
        └── model.py       # Model architecture
```

## 📊 Experiment Tracking
- Training progress visualization and metrics available on [W&B Dashboard](https://wandb.ai/miladnassiri92-topnetwork/covid-xray-classification/runs/16vcktjk)
![W&B Dashboard](https://github.com/miladnasiri/Covid-19-Radiology-Detection/blob/6dab2fe6cedc18215427aee91594a9b6b4809720/Experiment%20Tracking.png)
## 🔍 Model Analysis
### Strengths
- High accuracy on COVID-19 detection (99% precision)
- Robust performance across all classes
- Fast inference time

### Use Cases
- Medical diagnosis support
- Rapid screening
- Research applications

## 📚 References & Citation
```bibtex
@article{rahman2021exploring,
  title={Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images},
  author={Rahman, T. and Khandakar, A. and Qiblawey, Y. and Tahir, A. and Kiranyaz, S. and Kashem, S.B.A. and Islam, M.T. and Maadeed, S.A. and Zughaier, S.M. and Khan, M.S. and Chowdhury, M.E.},
  journal={Computers in Biology and Medicine},
  year={2021}
}
```

## 👤 Author
**Milad Nasiri**
- GitHub: [@miladnasiri](https://github.com/miladnasiri)
- LinkedIn: [Milad Nasiri](https://www.linkedin.com/in/miladnasiri/)

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

