"""
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""

from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init

__all__ = ['xception']

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):

    def __init__(self, num_classes=1000):

        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = nn.ReLU(inplace=True)(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def xception(num_classes=1000, pretrained='imagenet'):
    model = Xception(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    model.last_linear = model.fc
    del model.fc
    return model

### Import packages
import json
import torch
import PIL
import copy
import time
import numpy as np
import seaborn as sns
from collections import OrderedDict
from torch import nn, optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import datasets, transforms, models
from PIL import Image

train_dir = "training"

from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

len_train = 14000
indices = list(range(14000))

train_idx = indices[:11200]
valid_idx = indices[11200:]

train_indices = SubsetRandomSampler(train_idx)
valid_indices = SubsetRandomSampler(valid_idx)

# Using random rotation, random resize and random flips (horizontal and vertical)
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Just resize and crop
val_test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(train_dir, transform=val_test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
train_batch = 64
valid_batch = 32
test_batch = 64

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch, shuffle=False, sampler=train_indices)
validloader = torch.utils.data.DataLoader(train_dataset, batch_size=valid_batch, shuffle=False, sampler=valid_indices)

print(len(validloader))
images, labels = next(iter(validloader))
print(images.shape)
print(labels.shape)


model = xception()
print(model)

for param in model.parameters():
    param.requires_grad = False

model.last_linear = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(2048, 1024)),
                                ('relu1', nn.ReLU()),
                                ('dropout1', nn.Dropout(p=0.5)),
                                ('fc2', nn.Linear(1024, 512)),
                                ('relu2', nn.ReLU()),
                                ('dropout1', nn.Dropout(p=0.7)),
                                ('fc3', nn.Linear(512, 4))
                                ]))

criterion = nn.MSELoss()
optimizer = optim.SGD(model.last_linear.parameters(),lr=0.01, momentum=0.7, weight_decay=0.0001)  # lr=0.01, momentum=0.9, weight_decay=0.0005
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.2,patience=5,mode='max',min_lr=1e-3)  # step_size=8, gamma=0.5




#########################################

every_step = 1
valid_loss_min = np.Inf
total_epochs = 50

for epoch in range(total_epochs):
    
    ######################
    ###### Training ######
    ######################
    
    
    
    training_loss = 0
    validation_loss = 0
    counter = 0
    model.train()
    
    for images, labels in trainloader:

        counter+=1
        print("Training Batch: ", counter)
        
        #images, labels = images.to(device), labels.to(device) 
        
        optimizer.zero_grad()
        
        output = model.forward(images)
        
        if counter*train_batch < 11200:
            labels = labels_ls[(counter-1)*train_batch:counter*train_batch] 
        else:
            labels = labels_ls[(counter-1)*train_batch:11200]
        
        loss = criterion(output, labels)
        
        loss.backward()
        
        optimizer.step()
        
        training_loss += loss.item()
        
        if counter == 20:
            break

        scheduler.step(loss)
        
    if epoch%every_step == 0:

        ######################
        ##### Validation ####
        #####################

        valid_counter = 0
        model.eval()

        for data, target in validloader:

            valid_counter += 1
            print("Validation Batch: ", valid_counter)

            #data, target = data.to(device), target.to(device)

            out = model.forward(data)

            if valid_counter*valid_batch+11200 <= 14000:
                target = labels_ls[(valid_counter-1)*valid_batch+11200:valid_counter*valid_batch+11200]
            else:
                target = labels_ls[(valid_counter-1)*valid_batch+11200:valid_counter*valid_batch+14000]

            valid_loss = criterion(out, target)

            validation_loss += valid_loss.item()

        # calculating average losses
        training_loss = training_loss/len(trainloader.dataset)
        validation_loss = validation_loss/len(validloader.dataset)

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, training_loss, validation_loss))

        # save model if validation loss has decreased
        if validation_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            validation_loss))
            torch.save(model.state_dict(), 'object_localisation.pt')
            valid_loss_min = validation_loss