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
#inference for classification

#pre-processing image to send to model

def process_image(image):

  image = Image.open(image)

  transform = transforms.Compose([
      transforms.Resize(255),
      transforms.CenterCrop(224), 
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
  ])

  to_np_array = transform(image).float()
  
  return to_np_array



#predicting for image
def predict_util(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # pre-processing Image
    img=process_image(image_path)
    img=img.float().unsqueeze_(0)

    #moving model to device = "cuda"
    model.to("cuda")

    #moving image to cuda 
    img = img.to("cuda")

    #prediction step:
    with torch.no_grad():

      logps=model(img)

    #loading probabilities and indices  
    ps = torch.exp(logps)
    top_p, top_idx = ps.topk(topk, dim=1)

    # sending data back to cpu 
    probs = top_p.cpu().numpy()[0]

    #maping idx to classes
    idx = top_idx.cpu().numpy()[0]

    idx_to_class = { i:c for c, i in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in idx]


    return probs , classes


