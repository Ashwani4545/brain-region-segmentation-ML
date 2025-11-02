
# classifier.py - simple encoder-based classifier using the UNet encoder blocks
import torch
import torch.nn as nn
from model import conv_block

class EncoderClassifier(nn.Module):
    def __init__(self, in_channels=5, features=[32,64,128,256], num_classes=2):
        super().__init__()
        # reuse encoder blocks similar to UNet
        self.enc1 = conv_block(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(features[2], features[3])
        # global pooling + classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features[3], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        out = self.gap(e4)
        out = self.classifier(out)
        return out
