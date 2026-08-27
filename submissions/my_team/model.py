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




####### Iteration 4: Double Convolution ########
class ModelArchitecture(nn.Module):
    def __init__(self, num_classes: int = 20):
        super().__init__()

        # Double Convolution
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                # extract basics
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                          padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                # lets do another conv to be more expressive- increasing receptive field
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                          padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

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
        



