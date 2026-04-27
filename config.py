import torch

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
text_size = 1014
batch_size = 16

epochs = 100
num_class = 4
lr = 1e-3
mmt = 0.9

