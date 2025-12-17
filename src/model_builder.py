# src/model_builder.py
# ==============================================================================
# MODULE: MODEL BUILDER (M3)
# Research Standard: Dynamic graph construction with dummy-pass size inference.
# ==============================================================================

import torch
import torch.nn as nn

class DynamicCNN(nn.Module):
    """
    A CNN built dynamically from a genotype list.
    """
    def __init__(self, genotype, ways, input_channels=1, image_size=28):
        super().__init__()
        self.genotype = genotype
        self.ways = ways
        
        # 1. Build the Feature Extractor (The "Body")
        self.features, output_channels = self._build_features(genotype, input_channels)
        
        # 2. Calculate Linear Input Size (The "Bridge")
        # Research Standard: Use a dummy forward pass to find the exact output size.
        # This handles any combination of pooling/stride without manual math.
        self.flattened_size = self._get_flattened_size(output_channels, image_size)

        # 3. Build the Classifier (The "Head")
        self.classifier = nn.Linear(self.flattened_size, ways)

    def _build_features(self, genotype, in_channels):
        """Iterates through the DNA and adds corresponding PyTorch layers."""
        layers = []
        current_channels = in_channels
        
        for gene in genotype:
            gene_type = gene[0]
            
            if gene_type == 'conv':
                # Gene: ('conv', out_channels, kernel, padding)
                out_channels, kernel, padding = gene[1], gene[2], gene[3]
                layers.append(nn.Conv2d(current_channels, out_channels, kernel, padding=padding))
                current_channels = out_channels
                
            elif gene_type == 'pool':
                # Gene: ('pool', kernel, stride)
                kernel, stride = gene[1], gene[2]
                layers.append(nn.MaxPool2d(kernel_size=kernel, stride=stride))
                
            elif gene_type == 'relu':
                layers.append(nn.ReLU())
                
            elif gene_type == 'batchnorm':
                layers.append(nn.BatchNorm2d(current_channels))
                
        return nn.Sequential(*layers), current_channels

    def _get_flattened_size(self, channels, image_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, image_size, image_size)
            dummy_output = self.features(dummy_input)
            # Flatten: [Batch, Channels, Height, Width] -> [Batch, Flat_Size]
            return dummy_output.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x

def build_model_from_genotype(genotype, ways):
    """Factory function to create the model."""
    # Omniglot is grayscale, so input_channels=1
    return DynamicCNN(genotype, ways, input_channels=1, image_size=28)