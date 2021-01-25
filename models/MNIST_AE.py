import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Here we define the autoencoder model
"""


class MNIST_AE(nn.Module):
    
    def __init__(self):
        super(MNIST_AE, self).__init__()
        #encoder
        self.e1 = nn.Linear(784,256)
        self.e2 = nn.Linear(256,128)
        
        #Latent View
        self.lv = nn.Linear(128,64)
        
        #Decoder
        self.d1 = nn.Linear(64,128)
        self.d2 = nn.Linear(128,256)
        
        self.output_layer = nn.Linear(256,784)
        
    def forward(self,x):
        x = F.relu(self.e1(x))
        x = F.relu(self.e2(x))
        
        x = F.relu(self.lv(x))
        
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        
        x = self.output_layer(x)
        return x
    
    def get_embedding(self, x):
        x = F.relu(self.e1(x))
        x = F.relu(self.e2(x))
        
        x = F.relu(self.lv(x))
        
        return x
    
    
class mnistClassifier(nn.Module):
    def __init__(self, MNIST_AE):
        super(mnistClassifier, self).__init__()
        self.encoder = MNIST_AE 
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.encoder.get_embedding(x)
        x = F.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x
    
    def get_embedding(self, x):
        x = self.encoder.get_embedding(x)
        x = F.relu(self.linear1(x))
        return x
