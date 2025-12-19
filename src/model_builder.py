# src/model_builder.py
import torch
import torch.nn as nn

# --- 1. THE EVOLVED MODEL (Dynamic) ---
class DynamicCNN(nn.Module):
    def __init__(self, genotype, ways, input_channels=1, image_size=28):
        super().__init__()
        self.genotype = genotype
        self.features, output_channels = self._build_features(genotype, input_channels)
        self.flattened_size = self._get_flattened_size(input_channels, image_size)
        self.classifier = nn.Linear(self.flattened_size, ways)

    def _build_features(self, genotype, in_channels):
        layers = []
        current_channels = in_channels
        for gene in genotype:
            gene_type = gene[0]
            if gene_type == 'conv':
                out_channels, kernel, padding = gene[1], gene[2], gene[3]
                layers.append(nn.Conv2d(current_channels, out_channels, kernel, padding=padding))
                current_channels = out_channels
            elif gene_type == 'pool':
                layers.append(nn.MaxPool2d(kernel_size=gene[1], stride=gene[2]))
            elif gene_type == 'relu':
                layers.append(nn.ReLU())
            elif gene_type == 'batchnorm':
                layers.append(nn.BatchNorm2d(current_channels))
        return nn.Sequential(*layers), current_channels

    def _get_flattened_size(self, channels, image_size):
        with torch.no_grad():
            dummy = torch.zeros(1, channels, image_size, image_size)
            out = self.features(dummy)
            return out.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def build_model_from_genotype(genotype, ways):
    return DynamicCNN(genotype, ways)

# --- 2. THE BASELINE MODEL (Standard Conv-4) ---
# This is the "Human Designed" benchmark we beat.
def build_baseline_cnn(ways, input_channels=1):
    return nn.Sequential(
        # Layer 1
        nn.Conv2d(input_channels, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Layer 2
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Layer 3
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Layer 4
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        # Classifier
        nn.Flatten(),
        nn.Linear(64, ways) # For 28x28 input, 4 pools result in 1x1 feature map
    )