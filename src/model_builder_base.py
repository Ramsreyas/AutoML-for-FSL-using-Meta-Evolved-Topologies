# src/model_builder.py
# This file contains the logic for Module 3: The Genotype-to-Phenotype Mapper.
# It also holds the baseline model.

import torch
from torch import nn

class BaselineCNN(nn.Module):
    """
    A standard 4-block Convolutional Neural Network.
    This class-based structure is more flexible and allows for dynamic
    calculation of the final linear layer's input size.
    """
    def __init__(self, ways, channels=1):
        super().__init__()
        # Feature extractor part
        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 28x28 -> 14x14

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 14x14 -> 7x7

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 7x7 -> 3x3 (with padding)

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 3x3 -> 1x1 (with padding)
        )
        
        # Calculate the flattened feature size automatically.
        # This is a robust practice.
        with torch.no_grad():
            dummy_input = torch.randn(1, channels, 28, 28)
            dummy_output = self.features(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).size(1)

        # Classifier part
        self.classifier = nn.Linear(self.flattened_size, ways)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten the features
        x = self.classifier(x)
        return x

def build_baseline_cnn(ways, channels=1):
    """Public function to create an instance of the BaselineCNN."""
    return BaselineCNN(ways, channels)

# This is the placeholder for our next big task (M2)
def build_model_from_genotype(genotype, ways, channels=1):
    """
    (Placeholder) Translates a genotype (list of tuples) into a PyTorch model.
    """
    print("WARNING: Using placeholder build_model_from_genotype. Returning baseline model.")
    # For now, it just returns the baseline as a default.
    return build_baseline_cnn(ways, channels)