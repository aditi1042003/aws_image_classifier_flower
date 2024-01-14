import torch
from torch import nn
from torchvision import models
from collections import OrderedDict

#function to create model
#pre defined for four type of models
#1) densenet121
#2) resnet50
#3) alexnet
#4) vgg16
def Network(arch='vgg16' , dropout=0.4 , hidden_layer=1024):
  #choosing model from small to big size
  if arch == 'densenet121':
    model = models.densenet121(pretrained = True)
    input_nodes = 1024
  elif arch == 'resnet50':
    model = models.resnet50(pretrained = True)
    input_nodes = 2048
  elif arch == 'alexnet':
    model = models.alexnet(pretrained = True)
    input_nodes = 9216
  elif arch =='vgg16':
    model = models.vgg16(pretrained = True)
    input_nodes = 25088


  # Turn off gradients for our model
  for param in model.parameters():
    param.require_grad = False

  #defining classifier
  classifier = nn.Sequential(OrderedDict([
        ('dropout', nn.Dropout(dropout)),
        ('fc1', nn.Linear(input_nodes, hidden_layer)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_layer, 256)),
        ('output', nn.Linear(256, 102)),
        ('softmax', nn.LogSoftmax(dim = 1))
    ]))
  #adding the layers to model
  if (arch=='resnet50'):
    model.fc=classifier
  else:
    model.classifier=classifier

  return model