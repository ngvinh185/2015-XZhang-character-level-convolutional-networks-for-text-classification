import torch.nn as nn
import torch.nn.functional as F
import torch

class CharCNN(nn.Module):
  def __init__(self, num_class):
    super().__init__()
    self.layer1 = nn.Sequential(
        nn.Conv1d(in_channels = 70, out_channels = 256, kernel_size=7),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=3)
    )
    self.layer2 = nn.Sequential(
        nn.Conv1d(in_channels = 256, out_channels = 256, kernel_size=7),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=3)
    )
    self.layer3 = nn.Sequential(nn.Conv1d(in_channels = 256, out_channels = 256, kernel_size=3), nn.ReLU())
    self.layer4 = nn.Sequential(nn.Conv1d(in_channels = 256, out_channels = 256, kernel_size=3), nn.ReLU())
    self.layer5 = nn.Sequential(nn.Conv1d(in_channels = 256, out_channels = 256, kernel_size=3), nn.ReLU())
    self.layer6 = nn.Sequential(
        nn.Conv1d(in_channels = 256, out_channels = 256, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=3)
    )
    self.fc = nn.Sequential(
        nn.Linear(in_features=8704, out_features=2048),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(in_features=2048, out_features=2048),
        nn.ReLU(),
        nn.Dropout(0.5),

        nn.Linear(in_features=2048, out_features=num_class)
    )
    self.apply(self._init_weights)
  def _init_weights(self, module):
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean = 0, std = 0.05)
      if module.bias is not None: nn.init.zeros_(module.bias)
  def forward(self, x):
    # x shape: (B, 70, 1014)
    x = self.layer1(x)   # (B, 256, 336)
    x = self.layer2(x)   # (B, 256, 110)
    x = self.layer3(x)   # (B, 256, 108)
    x = self.layer4(x)   # (B, 256, 106)
    x = self.layer5(x)   # (B, 256, 104)
    x = self.layer6(x)   # (B, 256,  34)
    x = torch.flatten(x, 1)  # (B, 8704)
    x = self.fc(x)
    return x
