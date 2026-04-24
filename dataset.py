from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pandas as pd
import torch
from  config import *

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+=<>()[]{} \n"
char2idx = {x: i for i, x in enumerate(alphabet)}

def text2onehot(text, features = 70, max_length = text_size):
  one_hot_tensor = torch.zeros((features, max_length))
  text = text.lower()[:max_length]
  for idx, c in enumerate(text):
    if c in char2idx:
      one_hot_tensor[char2idx[c]][idx] = 1
  return one_hot_tensor

def handle_data(df):
  class dataset(Dataset):
    def __init__(self, texts, labels):
      self.texts = texts
      self.labels = labels
    def __len__(self):
      return len(self.texts)
    def __getitem__(self, index):
      x = text2onehot(self.texts[index])
      y = self.labels[index]
      return x, y
  data = dataset(texts = df['text'].tolist(), labels = df['label'].tolist())
  return data


ds = load_dataset("fancyzhx/ag_news")
df_train = pd.DataFrame(ds['train'])
df_dev = pd.DataFrame(ds['test'])


train_data = handle_data(df_train)
dev_data = handle_data(df_dev)
train_data_loader = DataLoader(train_data, batch_size, shuffle = True, num_workers=4)
dev_data_loader = DataLoader(dev_data, batch_size, num_workers=4)

