import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Classifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, out_features=num_classes)
        )
        # Freeze the base model parameters
        # for param in self.model.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        return self.model(x)

