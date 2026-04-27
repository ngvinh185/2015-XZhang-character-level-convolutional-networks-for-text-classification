from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pandas as pd
import torch
from  config import *

def dele(data, p = 0.1):
  words = data.split()
  a = torch.randn(len(words))
  a_norm = (a - a.min()) / (a.max() - a.min())
  new_words = []
  new_words += [word for idx, word in enumerate(words) if a_norm[idx] >= p]
  return " ".join(new_words)

def swap(data):
  a = torch.randint(1, 10, (1,))
  words = data.split()
  n = len(words)
  for _ in range(a):
    i1, i2 = torch.randint(0, n, (2,))
    words[i1], words[i2] = words[i2], words[i1]
  return " ".join(words)

def augment_data(data):
  choice = torch.randint(1, 3, (1,))
  if choice == 1:
    new_data = dele(data)
  else: new_data = swap(data)
  return new_data

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
      x = augment_data(self.texts[index])
      x = text2onehot(x)
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

