import torch
import torch.nn as nn
from torchvision import models

# simple baseline CNN
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super(BaselineCNN, self).__init__()
        num_classes = int(num_classes)
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        
        flattened_size = int(128 * 28 * 28)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=float(dropout_rate)),
            nn.Linear(flattened_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# Model factory
    
def get_model(model_name, config):
    # Wyciągamy wartości, podając domyślne, jeśli ich nie ma w configu
    num_classes = 10 
    dropout = config.get("dropout", 0.0)

    if model_name == "baseline_cnn":
        # PRZEKAZUJEMY WARTOŚCI, A NIE CAŁY SŁOWNIK 'config'
        return BaselineCNN(num_classes=num_classes, dropout_rate=dropout)
    
    elif "efficientnet_b0" in model_name:
        is_pretrained = "pretrained" in model_name
        weights = 'IMAGENET1K_V1' if is_pretrained else None
        
        model = models.efficientnet_b0(weights=weights)
        
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=float(dropout), inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )
        return model
    
    else:
        raise ValueError(f"Nieznany model: {model_name}")