import torch 
from Network import Network
# loading model
def load_model_util(path):
  checkpoint = torch.load(path)
  architecture = checkpoint['architecture']
  lr = checkpoint['learning_rate']
  hidden_layer = checkpoint['hidden_layer']
  device = checkpoint['device']
  epochs = checkpoint['epochs']
  state_dict = checkpoint['state_dict']
  class_to_idx = checkpoint['class_to_idx']
  dropout=checkpoint['dropout']

  model= Network(architecture, dropout, hidden_layer)
  model.class_to_idx = class_to_idx
  model.load_state_dict(state_dict)

  return model