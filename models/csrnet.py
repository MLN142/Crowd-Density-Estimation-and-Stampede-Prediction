import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CSRNet(nn.Module):
    def __init__(self, load_pretrained=True, batch_norm=True):
        super(CSRNet, self).__init__()

        # --- Frontend: Based on VGG-16 ---
        if batch_norm:
            vgg = models.vgg16_bn(pretrained=load_pretrained)
            self.frontend = nn.Sequential(*list(vgg.features.children())[:33])  # Up to conv4_3 with BN
        else:
            vgg = models.vgg16(pretrained=load_pretrained)
            self.frontend = nn.Sequential(*list(vgg.features.children())[:23])  # Up to conv4_3 without BN

        # --- Backend: Dilated convolutions ---
        if batch_norm:
            self.backend = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=1),
                nn.ReLU(inplace=True)
            )
        else:
            # Top View model uses 3x3 for the last bottleneck layer
            self.backend = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        # --- Output layer: 1-channel density map ---
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        # Initialize weights for backend and output
        self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return F.relu(x)  # Prevent negative values in density map

    def _initialize_weights(self):
        for m in self.backend.children():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)
