import torch

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(3))
text_size = 1014
batch_size = 16

epochs = 100
num_class = 4
lr = 1e-3
mmt = 0.9

