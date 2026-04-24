import os
from config import *
from model import *
from utils import *
from dataset import *
cnt = [0] * num_class
count = [0] * num_class
model = CharCNN(num_class=num_class).to(device)
if os.path.exists('checkpoint.pth'):
  load_model(model, optimizer = None, scheduler=None, path='checkpoint.pth')
  for x, y in dev_data_loader:
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
      outputs = model(x)
      predicted = torch.argmax(outputs, 1)
      for idx, x in enumerate(predicted.cpu().numpy()):
        count[y[idx].cpu().numpy()] += 1
        if x != y[idx].cpu().numpy():
          cnt[y[idx].cpu().numpy()] += 1
  print(cnt)
  print(count)
      