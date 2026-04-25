import os
from config import *
from model import *
from utils import *
from dataset import *
topics = ['World', 'Sport', 'Bussiness', 'Sci/Tech']
cnt = [0] * num_class
count = [0] * num_class
model = CharCNN(num_class=num_class).to(device)
if os.path.exists('checkpoint.pth'):
  load_model(model, optimizer = None, scheduler=None, path='checkpoint.pth')
  for x, y in dev_data_loader:
    # print(x.shape)
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
      outputs = model(x)
      predicted = torch.argmax(outputs, 1)
      predicted_np = predicted.cpu().numpy()
      y_np = y.cpu().numpy()
      for idx, xi in enumerate(x):
        print(xi.shape)
        if y_np[idx] != predicted_np[idx]:
          with open("output.txt", "a") as f:
            print(decode(xi), topics[y_np[idx]], topics[predicted_np[idx]], file = f)
            print('-----------------------------------', file = f)
      for idx, x in enumerate(predicted.cpu().numpy()):
        # print(x)
        
        count[y[idx].cpu().numpy()] += 1
        if x != y[idx].cpu().numpy():
          cnt[y[idx].cpu().numpy()] += 1
  print(cnt)
  print(count)
      