import torch
import torch.nn as nn

"""
#########iteration 1:Architecture: 2-Block Convolutional Neural Network (CNN)###########
class ModelArchitecture(nn.Module):
    def __init__(self, num_classes: int = 20):
        super().__init__()
        # Block 1: Input (3 channels) -> 16 channels
        # Shape goes from [batch, 3, 224, 224] -> [batch, 16, 112, 112]
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: 16 channels -> 32 channels
        # Shape goes from [batch, 16, 112, 112] -> [batch, 32, 56, 56]
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classifier: Flatten the 3D tensor and map to 20 classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 32 channels * 56 spatial height * 56 spatial width = 100,352 features
            nn.Linear(100352, num_classes)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        logits = self.classifier(x)

        return logits

"""




####### Iteration 2:4-block CNN featuring Batch Normalization, Dropout, and Global Average Pooling ########

class ModelArchitecture(nn.Module):
    def __init__(self, num_classes: int = 20):
        super().__init__()

        # Helper function to keep our code clean and DRY (Don't Repeat Yourself)
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                          padding=1),
                nn.BatchNorm2d(out_channels),
                # Normalizes the batch to train faster and stabler
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        # Feature Extractor: 4 Blocks to capture deeper, complex patterns
        # Input shape: [batch, 3, 224, 224]
        self.features = nn.Sequential(
            conv_block(3, 32),  # Output: [batch, 32, 112, 112]
            conv_block(32, 64),  # Output: [batch, 64, 56, 56]
            conv_block(64, 128),  # Output: [batch, 128, 28, 28]
            conv_block(128, 256)  # Output: [batch, 256, 14, 14]
        )

        # Classifier
        self.classifier = nn.Sequential(
            # AdaptiveAvgPool2d automatically averages the spatial dimensions down to 1x1.
            # This completely removes the need for hardcoded math like '100352'.
            # Output becomes: [batch, 256, 1, 1]
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            # Dropout randomly zeroes out 50% of the neurons during training.
            # This is a regularization technique to prevent overfitting.
            nn.Dropout(p=0.5),

            # Now we only map 256 features to 20 classes. Much cleaner!
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)

        return logits
        



