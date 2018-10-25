import argparse
import torch
#import progressbar
import torch.nn.functional as F
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
from PIL import Image
from collections import OrderedDict
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('Image Classifier')
parser.add_argument('data_dir', action='store',help='data storage path')
parser.add_argument('--save_dir', action='store',dest='save_dir',help='directory to save')
parser.add_argument('--learning_rate', action='store',dest='learning_rate', type=float, help='learning rate')
parser.add_argument('--hidden_units', action='store',dest='hidden_units', type=int, help='# of hidden units')
parser.add_argument('--epochs', action='store',dest='epochs', type=int, help='# of epochs')
parser.add_argument('--arch', action='store',dest='arch',help='NN architecture')
parser.add_argument('--gpu', action='store_true',dest='gpu',default=False,help='add this argument to use gpu')
params=vars(parser.parse_args())
data_dir, save_dir, learning_rate, hidden_units, epochs, arch, gpu=params['data_dir'], params['save_dir'], params['learning_rate'], params['hidden_units'], params['epochs'], params['arch'], params['gpu']

if save_dir==None:
    save_dir=''
if epochs==None:
    epochs=3
if hidden_units==None:
    hidden_units=1024
if learning_rate==None:
    learning_rate=0.001
#else:
#    learning_rate=float(learning_rate)
device='cpu'
if gpu:
    device='cuda'
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'    

traindata_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

data_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
image_datasets =[datasets.ImageFolder(train_dir, transform=traindata_transforms), 
                 datasets.ImageFolder(valid_dir, transform=data_transforms),
                 datasets.ImageFolder(test_dir, transform=data_transforms)]
dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
               torch.utils.data.DataLoader(image_datasets[1], batch_size=64),
               torch.utils.data.DataLoader(image_datasets[2], batch_size=64)]
num_classes=len(image_datasets[0].classes)
#load model with given architecture. By default use VGG19 with batchnorm
if arch==None:
    arch='vgg19_bn'
model=getattr(models,arch)(pretrained=True)

for param in model.parameters():
    param.requires_grad=False
in_features=model.classifier[0].weight.size()[1]

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(in_features, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, num_classes)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

active=[param for param in model.parameters() if param.requires_grad==True]
loss_function=nn.NLLLoss()
optimizer=optim.Adam(model.classifier.parameters(),lr=learning_rate)


model.to(device)
#print(model)

for epoch in range(epochs):
    running_loss=0
    examples_total=0
    model.train()
    for inputs, labels in dataloaders[0]:
        inputs, labels=inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        out=model(inputs)
        loss=loss_function(out,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        examples_total+=labels.size(0)
    print('epoch='+str(epoch)+', average training loss: '+str(running_loss/examples_total))
    running_loss=0
    examples_total=0
    correct=0
    model.eval()
    for inputs, labels in dataloaders[1]:
        inputs, labels=inputs.to(device), labels.to(device)
        with torch.no_grad():
            out=model(inputs)
            _, predicted = torch.max(out.data, 1)
            correct += (predicted == labels).sum().item()
            loss=loss_function(out,labels)
            running_loss+=loss.item()
            examples_total+=labels.size(0)
    print('epoch='+str(epoch)+', average validation loss: '+str(running_loss/examples_total))
    print('epoch='+str(epoch)+', validation accuracy: '+str(correct/examples_total))
    
try:
    os.makedirs(save_dir)    
except:
    pass
model.class_to_idx=image_datasets[0].class_to_idx
torch.save(model,save_dir+'model.pt')
