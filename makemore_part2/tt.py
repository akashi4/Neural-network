import torch
g = torch.Generator().manual_seed(42)

tensor = torch.empty((2, 2, 2), dtype=torch.float32)
# .uniform_(generator=g)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
torch.nn.init.xavier_uniform_(tensor)
print(tensor)
