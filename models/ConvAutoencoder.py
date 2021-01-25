import os
import copy
import scipy
import pickle
import numpy as np
from PIL import Image

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

# capacity = 16
imgSize = 256

class aeEncoder(nn.Module):
    def __init__(self, capacity, channel):
        super(aeEncoder, self).__init__()
        self.c = capacity
        self.channel = channel
        self.conv1 = nn.Conv2d(in_channels=self.channel, out_channels=self.c, kernel_size=4, stride=2, padding=1) # out: c x 256 x 256
        self.conv2 = nn.Conv2d(in_channels=self.c, out_channels=self.c*2, kernel_size=4, stride=2, padding=1) # out: c x 128 x 128
        self.conv3 = nn.Conv2d(in_channels=self.c*2, out_channels=self.c*4, kernel_size=4, stride=2, padding=1) # out: c x 64 x 64
        self.conv4 = nn.Conv2d(in_channels=self.c*4, out_channels=self.c*8, kernel_size=4, stride=2, padding=1) # out: c x 32 x 32
        self.conv5 = nn.Conv2d(in_channels=self.c*8, out_channels=self.c*16, kernel_size=4, stride=2, padding=1) # out: c x 16 x 16
            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        return x
    
class aeDecoder(nn.Module):
    def __init__(self, capacity, channel):
        super(aeDecoder, self).__init__()
        self.c = capacity
        self.channel = channel
        self.conv5 = nn.ConvTranspose2d(in_channels=self.c*16, out_channels=self.c*8, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(in_channels=self.c*8, out_channels=self.c*4, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(in_channels=self.c*4, out_channels=self.c*2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=self.c*2, out_channels=self.c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=self.c, out_channels=self.channel, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = x.view(x.size(0), self.c*16, (imgSize//32), (imgSize//32)) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x

class Autoencoder(nn.Module):
    def __init__(self, capacity, channel):
        super(Autoencoder, self).__init__()
        self.encoder = aeEncoder(capacity, channel)
        self.decoder = aeDecoder(capacity, channel)
    
    def forward(self, x):
        encoded = self.encoder(x)
        x_recon = self.decoder(encoded)
        return x_recon

    def get_embedding(self, x):
        emb = self.encoder(x)
        return emb

    
class aeClassifier(nn.Module):
    def __init__(self, Autoencoder, capacity):
        super(aeClassifier, self).__init__()
        c = capacity
        self.encoder = Autoencoder
        self.conv = nn.Conv2d(in_channels=c*16, out_channels=c*32, kernel_size=4, stride=2, padding=1) 
        self.linear1 = nn.Linear(512*4*4, 512)
        self.linear2 = nn.Linear(512, 1)
        
    def forward(self, x):
        x = self.encoder.get_embedding(x)
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x
    
    def get_embedding(self, x):
        x = self.encoder.get_embedding(x)
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        
        return x
    
