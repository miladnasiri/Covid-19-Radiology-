import torch
import torch.nn as nn
import timm

class CovidClassifier(nn.Module):
    def __init__(self, num_classes=4, model_name='efficientnet_b0'):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            drop_rate=0.3,
            drop_path_rate=0.2
        )
        
        # Add custom head
        num_features = self.model.get_classifier().in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
