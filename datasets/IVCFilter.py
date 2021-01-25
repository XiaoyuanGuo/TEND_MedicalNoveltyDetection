import os
import torch
import random
import numpy as np
from PIL import Image

from matplotlib import cm
from wand.image import Image as IImage
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])

from torchvision import transforms
from torch.utils.data.dataset import Dataset


class IVCFilter_Dataset(Dataset):
    
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        self.transformations = transforms.Compose([
                                     transforms.Resize((256,256)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(0.6188, 0.0816)
                                    ])
        
    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert("L")
        if self.transformations != None:
            img = self.transformations(img)
        return img, self.targets[idx]
    
    def __len__(self): 
        return len(self.data)
    

class IVCFilterTfm_Dataset(Dataset):
    def __init__(self, data, targets, transformations=None):

        self.data = data
        self.targets = targets
        self.transformations = transforms.Compose([
                                     transforms.Resize((256,256)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(0.6188, 0.0816)
                                    ])
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert("L")
        Timg = img
       
        pick = random.choice(range(0,9))
        
        if pick == 0:  
            with IImage(filename=self.data[idx]) as timg:
                timg.virtual_pixel = 'transparent'
                timg.distort('barrel', (0.2, 0.0, 0.0, 1.0))
                Timg = np.array(timg)
        elif pick == 1:
            with IImage(filename=self.data[idx]) as timg:
                timg.virtual_pixel = 'background'
                arguments = (0, 0, 20, 60, 90, 0, 70, 63, 0, 90, 5, 83, 90, 90, 85, 88)
                timg.distort('perspective', arguments)
                Timg = np.array(timg)
        elif pick == 2:
            with IImage(filename=self.data[idx]) as timg:
                timg.distort('arc', (95, ))
                Timg = np.array(timg)
        elif pick == 3:
            with IImage(filename=self.data[idx]) as timg:
                timg.distort('polar', (145, )) 
                Timg = np.array(timg)
        elif pick == 4:
            with IImage(filename=self.data[idx]) as timg:
                timg.background_color = timg[70, 46]
                timg.virtual_pixel = 'tile'
                timg.distort('arc', (60, ))
                Timg = np.array(timg)
        elif pick == 5:
            with IImage(filename=self.data[idx]) as timg:
                timg.virtual_pixel = 'tile'
                args = (0, 0, 30, 60, 140, 0, 110, 60, 0, 92, 2, 90, 140, 92, 138, 90)
                timg.distort('perspective', args)
                Timg = np.array(timg)
        elif pick == 6:
            with IImage(filename=self.data[idx]) as timg:
                timg.artifacts['distort:viewport'] = '300x200+50+50'
                args = (0, 0, 30, 60, 140, 0, 110, 60, 0, 92, 2, 90, 140, 92, 138, 90)
                timg.distort('perspective', args)
                Timg = np.array(timg)
        elif pick == 7:
            with IImage(filename=self.data[idx]) as timg:
                timg.virtual_pixel = 'background'
                args = (
                    10, 10, 15, 15,  # Point 1: (10, 10) => (15,  15)
                    139, 0, 100, 20, # Point 2: (139, 0) => (100, 20)
                    0, 92, 50, 80    # Point 3: (0,  92) => (50,  80)
                )
                timg.distort('affine', args)
                Timg = np.array(timg)
        else:
            with IImage(filename=self.data[idx]) as timg:
                timg.virtual_pixel = 'background'
                rotate = Point(0.1, 0)
                scale = Point(0.7, 0.6)
                translate = Point(5, 5)
                args = (scale.x, rotate.x, rotate.y,scale.y, translate.x, translate.y)
                timg.distort('affine_projection', args)
                Timg = np.array(timg)

        Timg = Image.fromarray(Timg).convert("L")
        if self.transformations != None:
            img = self.transformations(img)
            Timg = self.transformations(Timg)
        return (img, Timg), (0.0, 1.0)

    def __len__(self): 
        return len(self.data)   
    
    
def get_ivc_anomaly_dataset(normal_class):
    
    path = "/data/IVC_Filter/Stanford_Dataset/"
    classes = os.listdir(path)
    
    class_info = []
    for c in classes:
        classpath = os.path.join(path, c) 
        tmpclass = os.listdir(classpath)
        if len(tmpclass) == 1:
            class_info.append(os.path.join(classpath, tmpclass[0]))
        else:
            class_info.append(classpath)
    train_class_info = []
    test_class_info = []

    for c in class_info:
        train_class_info.append(os.path.join(c, "Training"))
        test_class_info.append(os.path.join(c, "Test"))
    
    stanford_train_data = train_class_info
    stanford_test_data = test_class_info

#     normal_class = [11]
    
    all_train_imgs = stanford_train_data
    all_test_imgs = stanford_test_data
    
#     trn_img = all_train_imgs[normal_class]
#     new_trn_img = [os.path.join(trn_img,x) for x in os.listdir(trn_img) if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg")]  
#     trn_test_img = all_test_imgs[normal_class]
#     new_trn_test_img = [os.path.join(trn_test_img,x) for x in os.listdir(trn_test_img) if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg")]

    new_trn_img = []
    for i in normal_class:
        trn_img = all_train_imgs[i]
        new_trn_img = new_trn_img + [os.path.join(trn_img,x) for x in os.listdir(trn_img) if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg")]
    
    new_trn_test_img = []
    for i in normal_class:
        trn_test_img = all_test_imgs[i]
        new_trn_test_img = new_trn_test_img + [os.path.join(trn_test_img,x) for x in os.listdir(trn_test_img) if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg")]
    
    new_tst_img = []    
    for i in range(0, len(all_train_imgs)):
        if i in normal_class:
            continue
        tst_img = all_train_imgs[i]
        new_tst_img = new_tst_img + [os.path.join(tst_img,x) for x in os.listdir(tst_img) if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg")]
    
    new_trn_lbl = [0]*len(new_trn_img)
    new_tst_lbl = [1]*len(new_tst_img) + [0]*len(new_trn_test_img)
    
    return new_trn_img, new_trn_lbl, new_tst_img+new_trn_test_img, new_tst_lbl    
