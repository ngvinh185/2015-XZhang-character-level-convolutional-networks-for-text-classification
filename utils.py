import os
import torch
import logging
from config import device
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
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
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