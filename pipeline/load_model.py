import torch

model = torch.load('model.pth')
model.to(0)

