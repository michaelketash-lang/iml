import torch
import torch.nn as nn


# Iteration 5: Double Convolution and spatial dropout ########
class ModelArchitecture(nn.Module):
    def __init__(self, num_classes: int = 20):
        super().__init__()

        # Double Convolution with Spatial Dropout
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                # extract basics
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                # lets do another conv to be more expressive- increasing receptive field
                nn.Conv2d(out_channels, out_channels, kernel_size=3,
                          stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                # Spatial Dropout (new try in iteration 5)
                nn.Dropout2d(p=0.1),

                # max pooling
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        # extract features still have 4 blocks but each block is double
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256)
        )

        # Classifier 20 classes
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)

        return logits
        



