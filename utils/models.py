import torch
import torch.nn as nn
from torchvision import models

# simple baseline CNN
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super(BaselineCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> 112x112
            
            # 2 convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> 56x56
            
            # 3 convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> 28x28
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Model factory
def get_model(model_name, num_classes=10, dropout_rate=0.0):
    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    
    elif model_name == "efficientnet_b0":
        # Model without pretrained weights
        model = models.efficientnet_b0(weights=None)
        
        # last layer modification
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )
        return model
    
    elif model_name == "efficientnet_b0_pretrained":
        # Model with pretrained weights
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # last layer modification
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )
        return model
    
    else:
        raise ValueError(f"Unknown model: {model_name}")