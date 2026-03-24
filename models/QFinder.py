"""
QFinder: CNN with Squeeze-and-Excitation blocks for substitution model classification.

This model classifies MSA into 7 substitution models:
- LG, WAG, JTT, Q.plant, Q.bird, Q.mammal, Q.pfam
"""

import torch
import torch.nn as nn


def conv1x1_bn_relu(in_channels: int, out_channels: int, padding='same', kernel_size=1) -> nn.Sequential:
    """
    Creates a 1x1 convolution block with BatchNorm and ReLU activation.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        padding: Padding mode (default: 'same')
        kernel_size: Kernel size (default: 1)
    
    Returns:
        Sequential module containing Conv2d, BatchNorm2d, and ReLU
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class SqueezeExcitation(torch.nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    
    This module adaptively recalibrates channelwise feature responses by explicitly
    modeling interdependencies between channels.
    
    Reference: Squeeze-and-Excitation Networks (Hu et al., 2018)
    """
    
    def __init__(
            self,
            input_channels: int,
            squeeze_channels: int,
            activation=torch.nn.ReLU,
            scale_activation=torch.nn.Sigmoid,
    ) -> None:
        """
        Args:
            input_channels: Number of input feature channels
            squeeze_channels: Number of channels in the squeeze layer
            activation: Activation function for the squeeze layer
            scale_activation: Activation function for the scale layer
        """
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input):
        """
        Computes channelwise scaling factors.
        
        Args:
            input: Input tensor of shape (B, C, H, W)
        
        Returns:
            Scaling factors of shape (B, C, 1, 1)
        """
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input):
        """
        Forward pass: applies channelwise scaling to input features.
        
        Args:
            input: Input tensor of shape (B, C, H, W)
        
        Returns:
            Scaled features of shape (B, C, H, W)
        """
        scale = self._scale(input)
        return scale * input


class QFinderModel(nn.Module):
    """
    Convolutional network with Squeeze-and-Excitation (SE) blocks for 
    substitution model classification (7 classes).
    
    Architecture:
    - Input: (B, 440, 25, 25) - QFinder features
    - 4 convolutional blocks, each followed by an SE block
    - Global average pooling
    - Fully connected layer for 7-class classification
    
    Output classes: LG, WAG, JTT, Q.plant, Q.bird, Q.mammal, Q.pfam
    """
    
    def __init__(self, num_classes: int = 7):
        """
        Args:
            num_classes: Number of substitution model classes
        """
        super().__init__()
        
        self.conv1 = conv1x1_bn_relu(440, 32)
        self.se1 = SqueezeExcitation(32, 64)

        self.conv2 = conv1x1_bn_relu(32, 64)
        self.se2 = SqueezeExcitation(64, 64)

        self.conv3 = conv1x1_bn_relu(64, 96)
        self.se3 = SqueezeExcitation(96, 64)

        self.conv4 = conv1x1_bn_relu(96, 32)
        self.se4 = SqueezeExcitation(32, 64)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.se1(x)
        x = self.conv2(x)
        x = self.se2(x)
        x = self.conv3(x)
        x = self.se3(x)
        x = self.conv4(x)
        x = self.se4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
