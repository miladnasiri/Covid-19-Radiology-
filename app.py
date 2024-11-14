from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import sys
import os
import timm
import numpy as np
import cv2

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import CovidClassifier

app = Flask(__name__)

# Correct class mapping based on your dataset
CLASS_MAPPING = {
    0: 'COVID',
    1: 'Lung_Opacity',
    2: 'Normal',
    3: 'Viral Pneumonia'
}

def preprocess_image(image):
    # Convert PIL Image to numpy array
    image = np.array(image)
    
    # Resize to model input size
    image = cv2.resize(image, (224, 224))
    
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Normalize pixel values
    image = image / 255.0
    
    # Apply standardization
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    return torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)

def load_model():
    model = CovidClassifier(num_classes=4)
    checkpoint = torch.load('outputs/best_model.pth', map_location=torch.device('cpu'))
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

print("Loading model...")
model = load_model()
print("Model loaded successfully")

@app.route('/')
def home():
    return '''
    <html>
        <head>
            <title>COVID-19 X-Ray Classification</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    max-width: 800px; 
                    margin: 0 auto; 
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container { 
                    text-align: center;
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                .result { 
                    margin-top: 20px; 
                    padding: 10px;
                    border-radius: 5px;
                }
                #imagePreview { 
                    max-width: 400px; 
                    margin: 10px auto;
                    border-radius: 5px;
                    box-shadow: 0 0 5px rgba(0,0,0,0.2);
                }
                .confidence { 
                    color: #2196F3;
                    font-weight: bold;
                }
                .probability-bar {
                    background-color: #e0e0e0;
                    border-radius: 5px;
                    margin: 5px 0;
                    height: 20px;
                }
                .probability-fill {
                    background-color: #2196F3;
                    height: 100%;
                    border-radius: 5px;
                    transition: width 0.5s ease-in-out;
                }
                .upload-btn {
                    background-color: #2196F3;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    margin: 10px;
                }
                .upload-btn:hover {
                    background-color: #1976D2;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>COVID-19 X-Ray Classification</h1>
                <p>Upload a chest X-ray image for classification</p>
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" id="file" accept="image/*" style="display: none">
                    <button type="button" class="upload-btn" onclick="document.getElementById('file').click()">
                        Choose File
                    </button>
                    <div id="imagePreview"></div>
                    <button type="button" class="upload-btn" onclick="predict()">Predict</button>
                </form>
                <div id="result" class="result"></div>
            </div>
            
            <script>
                function predict() {
                    const fileInput = document.getElementById('file');
                    if (!fileInput.files[0]) {
                        alert('Please select an image first');
                        return;
                    }
                    
                    const formData = new FormData();
                    formData.append('file', fileInput.files[0]);
                    
                    document.getElementById('result').innerHTML = 'Processing...';
                    
                    fetch('/predict', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        let html = `<h2>Prediction Results:</h2>
                                  <p>Predicted Class: <strong>${data.class}</strong></p>
                                  <p>Confidence: <span class="confidence">${(data.confidence * 100).toFixed(2)}%</span></p>
                                  <h3>All Probabilities:</h3>`;
                        
                        for (const [label, prob] of Object.entries(data.probabilities)) {
                            const percentage = (prob * 100).toFixed(2);
                            html += `
                                <p>${label}: ${percentage}%</p>
                                <div class="probability-bar">
                                    <div class="probability-fill" style="width: ${percentage}%"></div>
                                </div>`;
                        }
                        
                        document.getElementById('result').innerHTML = html;
                    })
                    .catch(error => {
                        document.getElementById('result').innerHTML = `Error: ${error}`;
                    });
                }
                
                document.getElementById('file').onchange = function(e) {
                    const preview = document.getElementById('imagePreview');
                    const file = e.target.files[0];
                    if (file) {
                        const reader = new FileReader();
                        reader.onload = function(e) {
                            preview.innerHTML = `<img src="${e.target.result}" style="max-width: 400px;">`;
                        }
                        reader.readAsDataURL(file);
                    }
                };
            </script>
        </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read and preprocess image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = preprocess_image(image)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        # Use correct class mapping
        prediction = {
            'class': CLASS_MAPPING[predicted_class],
            'confidence': float(probabilities[0][predicted_class]),
            'probabilities': {
                CLASS_MAPPING[i]: float(prob)
                for i, prob in enumerate(probabilities[0].tolist())
            }
        }

        return jsonify(prediction)

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
