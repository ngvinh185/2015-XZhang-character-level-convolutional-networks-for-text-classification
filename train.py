import torch
from config import *
from model import CharCNN
from dataset import train_data_loader, dev_data_loader
import torch.nn as nn
import torch.optim as optim
from utils import *
import os
# import time
# start = time.time()
model = CharCNN(num_class=num_class).to(device)
print(f'Day la {device}')
cel = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr = lr,
    momentum=mmt, 
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=20, gamma=0.1
)

train_lossi = []
dev_lossi = []
train_accuracyi = []
dev_accuracyi = []
logger = get_logger()

start_epoch = 0


cnt = 0
min_loss = 10
for epoch in range(start_epoch, epochs):
  model.train()
  print(f'epoch: {epoch + 1}')
  dev_loss_epoch = 0
  dev_accuracy_epoch = 0
  train_loss_epoch = 0
  train_accuracy_epoch = 0
  total_train = 0
  total_dev = 0
  for x, y in train_data_loader:
    x = x.to(device)
    y = y.to(device)
    output = model(x)
    # end = time.time()

    # batch_time = end - start
    # print(f"1 batch mất: {batch_time:.2f} giây")
    # break
  
    optimizer.zero_grad()
    loss = cel(output, y)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
      train_accuracy_epoch += (torch.argmax(output, dim = 1) == y).sum().item()
      train_loss_epoch += loss.item() * x.shape[0]
      total_train += x.shape[0]
  # break
  train_accuracy_epoch /= total_train
  train_loss_epoch /= total_train
  
  model.eval()
  with torch.no_grad():
    for x, y in dev_data_loader:
      x = x.to(device)
      y = y.to(device)
      output = model(x)
      loss = cel(output, y)
      dev_accuracy_epoch += (torch.argmax(output, dim = 1) == y).sum().item()
      dev_loss_epoch += loss.item() * x.shape[0]
      total_dev += x.shape[0]

    dev_accuracy_epoch /= total_dev
    dev_loss_epoch /= total_dev
  if dev_loss_epoch < min_loss:
    min_loss = dev_loss_epoch
    torch.save({
      'epoch': epoch + 1,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'scheduler_state_dict': scheduler.state_dict(),
      'loss': dev_loss_epoch,
  }, 'checkpoint.pth')
  if dev_loss_epoch < min_loss + 0.2:
    cnt = 0
  else: cnt += 1
  if cnt >= 7: break
  train_lossi.append(train_loss_epoch)
  dev_lossi.append(dev_loss_epoch)
  train_accuracyi.append(train_accuracy_epoch)
  dev_accuracyi.append(dev_accuracy_epoch)

  scheduler.step()


  # print(f'Train Loss in epoch {epoch + 1} = {train_loss_epoch}')
  # print(f'Dev Loss in epoch {epoch + 1} = {dev_loss_epoch}')
  # print(f'Train Accuracy in epoch {epoch + 1} = {train_accuracy_epoch}')
  # print(f'Dev Accuracy in epoch {epoch + 1} = {dev_accuracy_epoch}')
  
  logger.info(f'Epoch {epoch + 1}: Train Loss = {train_loss_epoch:.4f}, Dev Loss = {dev_loss_epoch:.4f}, Train Accuracy = {train_accuracy_epoch:.4f}, Dev Accuracy = {dev_accuracy_epoch:.4f}')