
import os
import cv2
import random
import scipy.misc
import numpy as np
from skimage import io
from PIL import Image as im

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.dataset import Dataset

from wand.image import Image as IImage
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])


class CIFAR10LabelDataset(Dataset):
    def __init__(self, data, ydata, transform):
        self.data = data
        self.ydata = ydata
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.ydata[idx]
        x = np.array(x)
        if self.transform!=None:
            x = self.transform(x)   
        return x, label
    

class CIFAR10TfmDataset(Dataset):
    def __init__(self, data, ydata, transform):
        self.data = data
        self.ydata = ydata
        self.transform = transform
       
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.ydata[idx]
        
        timg = im.fromarray(np.array(x))
        timg.save("test_cifar.jpeg")

        pick = random.choice(range(0,8))
        if pick == 0:  
            with IImage(filename='test_cifar.jpeg') as timg:
                timg.virtual_pixel = 'transparent'
                timg.distort('barrel', (1.2,1.5, 0.6, 1.0))
                timg.resize(32, 32)
                Timg = np.array(timg)
        elif pick == 1:
            with IImage(filename='test_cifar.jpeg') as timg:
                timg.virtual_pixel = 'background'
                arguments = (
                    10, 10, 15, 25, 
                    13, 9, 20, 27, 
                    12, 15, 19, 17
                )
                timg.distort('perspective', arguments)
                timg.resize(32, 32)
                Timg = np.array(timg)
        elif pick == 2:
            with IImage(filename='test_cifar.jpeg') as timg:
                timg.distort('polar', (45, )) 
                timg.resize(32, 32)
                Timg = np.array(timg)
        elif pick == 3:
            with IImage(filename='test_cifar.jpeg') as timg:
                timg.virtual_pixel = 'tile'
                timg.distort('arc', (60, ))
                timg.resize(32, 32)
                Timg = np.array(timg)
        elif pick == 4:
            with IImage(filename='test_cifar.jpeg') as timg:
                timg.virtual_pixel = 'tile'
                args = (10, 10, 24, 25)
                timg.distort('perspective', args)
                timg.resize(32, 32)
                Timg = np.array(timg)
        elif pick == 5:
            with IImage(filename='test_cifar.jpeg') as timg:
                timg.virtual_pixel = 'background'
                args = (9, 12, 21, 19, 
                    14, 12, 13, 9)
                timg.distort('perspective', args)
                timg.resize(32, 32)
                Timg = np.array(timg)
        elif pick == 6:
            with IImage(filename='test_cifar.jpeg') as timg:
                timg.virtual_pixel = 'background'
                args = (
                    10, 10, 15, 15, 
                    13, 19, 10, 26
                    )
                timg.distort('affine', args)
                timg.resize(32, 32)
                Timg = np.array(timg)
        else:
            with IImage(filename='test_cifar.jpeg') as timg:
                timg.virtual_pixel = 'background'
                rotate = Point(0.1, 0)
                scale = Point(0.3, 0.6)
                translate = Point(5, 5)
                args = (scale.x, rotate.x, rotate.y,scale.y, translate.x, translate.y)
                timg.distort('affine_projection', args)
                timg.resize(32, 32)
                Timg = np.array(timg)
                
        if Timg.shape[2] == 4:
            Timg = np.delete(Timg, 3, 2)

        if self.transform!=None:
            x = self.transform(x)  
            Timg = self.transform(Timg)
        return (x, Timg), (0.0, 1.0)
    

    
def get_cifar10_data(normal_class):
    
    CIFAR10_PATH = "/home/jupyter-sophie/DATA/"
    # extract data and targets
    train_data = datasets.CIFAR10(root=CIFAR10_PATH, train=True, download=False)
    x_train, y_train=train_data.data,train_data.targets
    test_data=datasets.CIFAR10(root=CIFAR10_PATH, train=False, download=False)
    x_test, y_test=test_data.data,test_data.targets

    outlier_classes = list(range(0, 10))

    for c in normal_class:
        outlier_classes.remove(c)
    normal_x_train = [] #train data
    normal_y_train = [] #train label
    outlier_x_test = [] #outlier data for final testing
    outlier_y_test = [] #outlier label for final testing
    for i in range(0, len(y_train)):
        if y_train[i] in normal_class:
            normal_x_train.append(x_train[i])
            normal_y_train.append(0)
        else:
            outlier_x_test.append(x_train[i])
            outlier_y_test.append(1)
            
    normal_x_val = [] #train data
    normal_y_val = [] #train label
    for i in range(0, len(y_test)):
        if y_test[i] in normal_class:
            normal_x_val.append(x_test[i])
            normal_y_val.append(0)
        else:
            outlier_x_test.append(x_test[i])
            outlier_y_test.append(1)
            
    return normal_x_train, normal_y_train, normal_x_val, normal_y_val, outlier_x_test, outlier_y_test
                
    
    
    
    
    
