cat >> README.md << 'EOL'

## 📊 Dataset Details

### COVID-19 Radiography Database
![Kaggle Award](https://img.shields.io/badge/Kaggle-Dataset%20Award%20Winner-blue)

This project utilizes the award-winning COVID-19 Radiography Database, a comprehensive collection of chest X-ray images created through international collaboration between:
- Qatar University, Doha, Qatar
- University of Dhaka, Bangladesh
- Medical professionals from Pakistan and Malaysia

### Dataset Evolution
The database has evolved through multiple stages:
```
Initial Release:
- COVID-19: 219 images
- Normal: 1,341 images
- Viral Pneumonia: 1,345 images

Current Version:
- COVID-19: 3,616 images
- Normal: 10,192 images
- Lung Opacity: 6,012 images
- Viral Pneumonia: 1,345 images
Total: 21,165 images
```

### Dataset Characteristics
- **Image Format**: PNG format
- **Resolution**: 299×299 pixels
- **Type**: Grayscale chest X-rays
- **Annotations**: Includes corresponding lung masks
- **Quality**: Medical-grade, verified images

### Class Distribution
```
Distribution Ratio (relative to smallest class):
- Normal: 7.58x
- Lung Opacity: 4.47x
- COVID-19: 2.69x
- Viral Pneumonia: 1.00x (base)
```

### Data Sources
The COVID-19 images were collected from multiple sources:
1. Padchest dataset (2,473 images)
2. German medical school (183 images)
3. SIRM, Github, Kaggle & Twitter (559 images)
4. Additional GitHub sources (400 images)

### Verification Process
- All images were verified by medical professionals
- Quality control measures were implemented
- Proper documentation and labeling
- Expert validation of classifications

### Dataset Access
- [Kaggle Dataset Link](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- Winner of the COVID-19 Dataset Award by Kaggle Community
- Regularly updated with new verified cases

### Data Preprocessing
For this project, we implemented the following preprocessing steps:
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

### Data Augmentation Strategy
To address class imbalance and improve model generalization:
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

### Citation
```bibtex
@article{rahman2021exploring,
  title={Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images},
  author={Rahman, T. and Khandakar, A. and Qiblawey, Y. and Tahir, A. and Kiranyaz, S. and Kashem, S.B.A. and Islam, M.T. and Maadeed, S.A. and Zughaier, S.M. and Khan, M.S. and Chowdhury, M.E.},
  journal={Computers in Biology and Medicine},
  year={2021}
}
```
EOL

git add README.md
git commit -m "Add comprehensive dataset documentation and analysis"
git push origin main
