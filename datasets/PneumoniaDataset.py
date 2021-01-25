import os
import cv2
import random
import numpy as np
import scipy.misc
import pandas as pd
from PIL import Image
from skimage import io
import pydicom as dicom

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from wand.image import Image as IImage
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])

"""
pneumonia dataset class, return image 
"""

imgSize = 256
train_tfms = transforms.Compose([
#     transforms.Resize((imgSize,imgSize)),
    transforms.ToTensor(),
    transforms.Normalize(0.4977, 0.2356)
])


class PneumoniaDataset(Dataset):
    def __init__(self, traindata, labels, transform=train_tfms):
        """
        datapath: dicom data path
        """
        self.transform = transform
        self.data = traindata
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.data[idx].endswith(".dcm"):
            dcm = dicom.read_file(self.data[idx])
            image = Image.fromarray(dcm.pixel_array)
        elif self.data[idx].endswith(".png") or self.data[idx].endswith(".jpg") or self.data[idx].endswith(".jpeg"):
            image = Image.open(self.data[idx]).convert("L")
            
#         if self.transform != None:
#             image = self.transform(image)
        image = image.resize((imgSize, imgSize))    
        return train_tfms(image), self.labels[idx]
    

class PneumoniaTfmDataset(Dataset):
    def __init__(self, traindata, transform=train_tfms):
        """
        datapath: dicom data path
        """
        self.transform = transform
        self.data = traindata
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.data[idx].endswith(".dcm"):
            dcm = dicom.read_file(self.data[idx])
            image = Image.fromarray(dcm.pixel_array)
        elif self.data[idx].endswith(".png") or self.data[idx].endswith(".jpg") or self.data[idx].endswith(".jpeg"):
            image = Image.open(self.data[idx]).convert("L")
        
        image = image.resize((imgSize, imgSize))
        t = np.expand_dims(np.array(image), axis = 2)
        t = np.concatenate((t, t, t), axis = 2)
        timg = Image.fromarray(t)
        timg.save("rsna_test.jpeg")

        pick = random.choice(range(0,8))

        if pick == 0:  
            with IImage(filename='rsna_test.jpeg') as timg:
                timg.virtual_pixel = 'transparent'
                timg.distort('barrel', (1.2, 0.5, 0.6, 1.0))
                timg.resize(imgSize, imgSize)
                Timg = np.array(timg)
        elif pick == 1:
            with IImage(filename='rsna_test.jpeg') as timg:
                timg.virtual_pixel = 'background'
                arguments = (
                    90, 100, 115, 215, 
                    113, 194, 201, 217, 
                    172, 155, 241, 257
                )
                timg.distort('perspective', arguments)
                timg.resize(imgSize, imgSize)
                Timg = np.array(timg)
        elif pick == 2:
            with IImage(filename='rsna_test.jpeg') as timg:
                timg.distort('polar', (555, )) 
                timg.resize(imgSize, imgSize)
                Timg = np.array(timg)
        elif pick == 3:
            with IImage(filename='rsna_test.jpeg') as timg:
                timg.virtual_pixel = 'tile'
                timg.distort('arc', (60, ))
                timg.resize(imgSize, imgSize)
                Timg = np.array(timg)
        elif pick == 4:
            with IImage(filename='rsna_test.jpeg') as timg:
                timg.virtual_pixel = 'tile'
                args = (80, 80, 24, 55)
                timg.distort('perspective', args)
                timg.resize(imgSize, imgSize)
                Timg = np.array(timg)
        elif pick == 5:
            with IImage(filename='rsna_test.jpeg') as timg:
                timg.virtual_pixel = 'background'
                args = (119, 112, 251, 169, 
                        84, 108, 183, 197)
                timg.distort('perspective', args)
                timg.resize(imgSize, imgSize)
                Timg = np.array(timg)
        elif pick == 6:
            with IImage(filename='rsna_test.jpeg') as timg:
                timg.virtual_pixel = 'background'
                args = (
                    210, 108, 195, 175, 
                    113, 119, 109, 236
                )
                timg.distort('affine', args)
                timg.resize(imgSize, imgSize)
                Timg = np.array(timg)
        else:
            with IImage(filename='rsna_test.jpeg') as timg:
                timg.virtual_pixel = 'background'
                rotate = Point(0.5, 0)
                scale = Point(0.8, 0.9)
                translate = Point(5, 5)
                args = (scale.x, rotate.x, rotate.y,scale.y, translate.x, translate.y)
                timg.distort('affine_projection', args)
                timg.resize(imgSize, imgSize)
                Timg = np.array(timg)

        Timg = Image.fromarray(Timg).convert("L") 
        image = train_tfms(image)
        Timg = train_tfms(Timg)
            
        return (image, Timg), (0.0, 1.0)


def get_rsna_data():
    
    PNEUMONIA_FILE_PATH = "/home/jupyter-sophie/CBIR/rsna/image_bbox_full.csv"
    all_df = pd.read_csv(PNEUMONIA_FILE_PATH)
    
    normal_data = all_df.loc[all_df["class"] == "Normal"]
    opacity_data = all_df.loc[all_df["class"] == "Lung Opacity"]
    other_data = all_df.loc[all_df["class"] == "No Lung Opacity / Not Normal"]
    
    normal_path = list(normal_data["path"])
    opacity_path = list(opacity_data["path"])
    other_path = list(other_data["path"])
    
    return normal_path, opacity_path, other_path
