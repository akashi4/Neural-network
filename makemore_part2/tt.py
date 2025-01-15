import torch

tensor = torch.ones((2, 2, 2), dtype=torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.nn.init.normal_(tensor)
print(tensor)
