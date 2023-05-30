from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

from kornia.core import Module, Tensor
from kornia.core.check import KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE

#from .kernels import _unpack_2d_ks, get_binary_kernel2d

import torch
#torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import cv2
import os
import numpy as np 
import skimage.io as io
import skimage.transform as trans
import math
from torch.nn.modules.utils import _pair, _quadruple
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import kornia.filters

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 4) #CLASStorch.nn.Conv2d(in_channels, out_channels(NO. OF FILTERS), kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        self.pool = nn.MaxPool2d(2, 2)  # nn.MaxPool2d(kernel size, stride )
        self.conv2 = nn.Conv2d(6, 16, 4) #CLASStorch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # self.conv2d_drop = nn.Dropout()
        self.fc1 = nn.Linear(16 * 4 * 4, 120) #CLASStorch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100) # we can chan  ge 120,84 but 16*4*4 and 14 must be fixed. 14 is no. of classes)
#for forward pass 
#net= Net()
#print(net)
    #def lprelu(xx, a, z):
        #if xx<=0:
            #xx=0
        #elif 0>xx<=a:
            #xx=xx
        #else:
            #xx=a+z(xx-a)
        #return xx
    

    #def find_medians(self, x, k):
        #patches = tf.compat.v1.extract_image_patches(
            #x, 
            #ksizes=[1, k, k, 1],
            #strides = [1, 1, 1, 1],
            #rates=[1, 1, 1, 1],
            #padding='SAME')
        #print("k=", k)
        #m_idx = int(k*k/2 + 1)
        #top, _ = tf.nn.top_k(patches, m_idx, sorted=True)
        #x = tf.slice(top, [0, 0, 0, m_idx-1], [-1, -1, -1, 1])
        #return torch.cuda.FloatTensor(Tensor.cpu((x))

    #def gaussian_filter(self, x):
       #transform = transforms.GaussianBlur(kernel_size=(15, 15), sigma=(0.1, 0.2))
        #x=transform(x)
        #return x

    
    def median(self,x):
        #y = kornia.filters.MedianBlur(5)
        #x=y(x)
        #return x
    
    def forward(self, x):
       x = self.median(x)
        #x = self.gaussian_filter(x)
        #x = self.find_medians(x, 3)
        x = self.pool(F.relu(self.conv1(x))) # x = self.pool(self.lprelu(self.conv1(x), 0.5, 1))
        # x = self.pool(F.relu(self.conv2d_drop(self.conv1(x))))
        #x = self.gaussian_filter(x)
        x = self.pool(F.relu(self.conv2(x)))
         # x = self.pool(self.lprelu(self.conv2(x), 0.5, 1))
        # x = self.pool(F.relu(self.conv2d_drop(self.conv2(x))))
        x = x.view(-1, 16 * 4 * 4) # flatten
        x = F.relu(self.fc1(x)) # x = self.lprelu(self.fc1(x), 0.5, 1)
        x = F.relu(self.fc2(x)) # x = self.lprelu(self.fc2(x), 0.5, 1)
        x = self.fc3(x) 
        return x
    
    
class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super(SoftmaxClassifier, self).__init__()
        self.fc = nn.Linear(28 * 28 * 3, 100, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_().mul_(0.005)

    def forward(self, x):
        x = x.view(-1, 28 * 28 * 3)
        x = self.fc(x)
        return x

