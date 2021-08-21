#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import cv2
from PIL import Image

# import Dataset & DataLoader 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
# import image process package
import torchvision
import torchvision.transforms as transforms

import os
import torch

trans = transforms.Compose([
    # define your own image preprocessing
    transforms.CenterCrop(5),
    # convert to tensor
    transforms.ToTensor()
])

class txt_dataset(Dataset):
    # override the init function
    def __init__(self, data_dir, label_dir, file_num):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.file_num = file_num
    #override the getitem function
    def __getitem__(self, index):
        data_name = os.path.join(self.data_dir, str(index+1)+".jpg")
        label_name = os.path.join(self.label_dir, str(index+1)+".txt")
        D = Image.open(data_name)
        D = trans(D)
        L = np.loadtxt(label_name)
        return D,L
    #override the len function    
    def __len__(self):
      return self.file_num
          

# declare training dataset
train_dataset = txt_dataset(data_dir="/home/U111syhuang/AI_Training_HW3/train_data/",label_dir="/home/U111syhuang/AI_Training_HW3/train_label/",file_num=9)

# declare testing dataset
test_dataset = txt_dataset(data_dir="/home/U111syhuang/AI_Training_HW3/test_data/", label_dir="/home/U111syhuang/AI_Training_HW3/test_label/", file_num=3)

# declare training dataloader
trainloader = DataLoader(train_dataset, shuffle=True, batch_size=1)

# declare testing dataloader
testloader = DataLoader(test_dataset, shuffle=True, batch_size=1)

print('train_data & label')
for index, data_package in enumerate(trainloader):
    train_data, train_label = data_package
    print('Type: ', type(train_data))
    print('Index: ',index+1)
    print('Data: ',train_data)
    print('Label:', train_label)

print('test_data & label')
for test_data, test_label in testloader:
    print('Data: ',test_data)
    print('Label:', test_label)

