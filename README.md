# COVID-19 X-Ray Classification

## Overview
Deep learning model for COVID-19 X-ray classification using PyTorch and EfficientNet.

## Results
- Training Accuracy: 97.38%
- Validation Accuracy: 95.89%
- W&B Dashboard: https://wandb.ai/miladnassiri92-topnetwork/covid-xray-classification/runs/16vcktjk

## Project Structure
```
├── data/
│   └── raw/COVID-19_Radiography_Dataset/
├── src/
│   ├── data/
│   ├── models/
│   └── utils/
├── outputs/
└── requirements.txt
```

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training
```bash
python src/train.py
```

## Evaluation
```bash
python src/evaluate.py
```

## Prediction
```bash
python src/predict.py path/to/xray.png
```
