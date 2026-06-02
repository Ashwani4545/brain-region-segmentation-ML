
# model.py - 2.5D UNet
import torch
import torch.nn as nn

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class UNet2p5D(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, features=[32,64,128,256]):
        super().__init__()
        self.enc1 = conv_block(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(features[2], features[3])

        self.up3 = nn.ConvTranspose2d(features[3], features[2], 2, stride=2)
        self.dec3 = conv_block(features[3], features[2])
        self.up2 = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.dec2 = conv_block(features[2], features[1])
        self.up1 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.dec1 = conv_block(features[1], features[0])

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)

        d3 = self.up3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        return torch.sigmoid(out)
