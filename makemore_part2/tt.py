import torch

tensor = torch.ones((2, 2, 2), dtype=torch.float32)
print(tensor)
torch.nn.init.normal_(tensor)
print(tensor)
