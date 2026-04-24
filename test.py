import torch
a = torch.randn(10000, 10000).to('cuda:3')
b = torch.randn(10000, 10000).to('cuda:3')
c = a @ b
