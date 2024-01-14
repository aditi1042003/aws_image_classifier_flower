#Imports here
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

#Importing functions created for predict.py
from load_model_util import load_model_util
from predict_util import predict_util

#not making different util file for command line arguments for predict file
parser = argparse.ArgumentParser()

parser.add_argument('--flowername', default='cat_to_name.json', help='enter the class name file')
parser.add_argument('--test_image', default='flowers/test/13/image_05745.jpg', help='enter test image location')
parser.add_argument('--checkpoint', default='checkpoint.pth', help='enter trained model filepath')
parser.add_argument('--topk', type=int, default=5, help='Allows user to enter the top "k" predictions suggested by the model.')
parser.add_argument('--device', default='cuda', type=str, help='Determines which model to run')

in_args=parser.parse_args()
#reading CLI arguments passed with predict.py
flowername=in_args.flowername
checkpoint=in_args.checkpoint
test_image=in_args.test_image
topk=in_args.topk
device=in_args.device

#importing classes name file
with open(flowername, 'r') as f:
    cat_to_name = json.load(f)
    

# loading model
model1 = load_model_util(checkpoint)
print(model1)
# def load_model(path):
#   checkpoint = torch.load(path)
#   architecture = checkpoint['architecture']
#   lr = checkpoint['learning_rate']
#   hidden_layer = checkpoint['hidden_layer']
#   device = checkpoint['device']
#   epochs = checkpoint['epochs']
#   state_dict = checkpoint['state_dict']
#   class_to_idx = checkpoint['class_to_idx']

#   model= Network(architecture, dropout, hidden_layer)
#   model.class_to_idx = class_to_idx
#   model.load_state_dict(state_dict)

#   return model



# loading model
# model1 = load_model('checkpoint.pth')
# print(model1)

#predicting for the image
probs, classes = predict_util(test_image,model1,topk)
print(probs)
print(classes)
class_name=[cat_to_name[i] for i in classes]
print(class_name)
