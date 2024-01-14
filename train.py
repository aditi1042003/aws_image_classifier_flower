# Imports here

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import json
import argparse
#Importing functions created for train.py
from get_input_args import get_input_args
from Network import Network
from train_util import train_util
from test_util import test_util

#Maps parser arguments to variables for ease of use later
in_args = get_input_args()

data_dir=in_args.data_dir
save_dir=in_args.save_dir
arch=in_args.arch
flowername=in_args.flowername
lr=in_args.learning_rate
hidden_layer=in_args.hidden_layer
device=in_args.device
epochs=in_args.epochs
dropout=in_args.dropout


#importing classes name file
with open(flowername, 'r') as f:
    cat_to_name = json.load(f)

    
# TODO: Define your transforms for the training, validation, and testing sets
data_transforms_training = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

#loading data
#data_dir = '/content/flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
data_transforms_testing_validation = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])



# TODO: Load the datasets with ImageFolder
#image_datasets =
train_data = datasets.ImageFolder(train_dir, transform=data_transforms_training)
valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms_testing_validation)
test_data = datasets.ImageFolder(test_dir, transform=data_transforms_testing_validation)


# TODO: Using the image datasets and the trainforms, define the dataloaders
#dataloaders =
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)



validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)


#creating model
model = Network(arch,dropout,hidden_layer)
print(model)


#loss criterion
criterion = nn.NLLLoss()
#optimizer
if (arch=='resnet50'):
  optimizer = optim.Adam(model.fc.parameters(), lr=lr)
else:
  optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

#training model
train_util(model=model, trainloader=trainloader,validloader=validloader, optimizer= optimizer , criterion=criterion, epochs=epochs , device=device)

#testing model
#test_util(testloader,model,device)

#saving checkpoint
model.class_to_idx = train_data.class_to_idx
checkpoint ={
    "architecture": arch,
    "learning_rate": lr,
    "hidden_layer": hidden_layer,
    'device': device,
    'epochs': epochs,
    'dropout': dropout,
    'state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx
}


torch.save(checkpoint , save_dir)
