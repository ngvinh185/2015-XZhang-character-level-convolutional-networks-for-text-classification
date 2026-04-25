import os
import torch
import logging
from config import device
from dataset import alphabet
def save_model(model, optimizer, scheduler, epoch, loss, path):
  torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
  }, path)
  
def load_model(model, optimizer, scheduler, path):
  checkpoint = torch.load(path, map_location=device)
  model.load_state_dict(checkpoint['model_state_dict'])
  
  if optimizer is not None: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  if scheduler is not None: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']
  return epoch, loss

def get_logger(logs_dir = 'logs'):
  os.makedirs(logs_dir, exist_ok = True)
  logging.basicConfig(
    level=logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
      logging.FileHandler(os.path.join(logs_dir, 'training.log')),
      logging.StreamHandler()
    ]
  )
  return logging.getLogger()


def decode(x):
  s = ""
  print(len(x))
  for i in range(1014):
    for j in range(70):
      if(x[j][i] == 1):
        s += alphabet[j]
        break
  return s 