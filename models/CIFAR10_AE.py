import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Here we define the autoencoder model
"""


class CIFAR10Autoencoder(nn.Module):
    def __init__(self):
        super(CIFAR10Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1)   # [batch, 3, 32, 32]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_embedding(self, x):
        encoded = self.encoder(x)
        return encoded
    
    
    
class CIFAR10AutoencoderCLS(nn.Module):
    def __init__(self, CIFAR10Autoencoder):
        super(CIFAR10AutoencoderCLS, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = CIFAR10Autoencoder
        self.conv = nn.Sequential(nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
            nn.ReLU(),
        )
        self.linear = nn.Linear(96*2*2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder.get_embedding(x)
        x = self.conv(x)
        x = self.linear(x.view(x.size(0),-1))
        x = self.sigmoid(x)
        return x
    
    def get_embedding(self, x):
        x = self.encoder.get_embedding(x)
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return x
