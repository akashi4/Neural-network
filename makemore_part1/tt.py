import torch
g = torch.Generator().manual_seed(45)
x1 = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0], [1, 0, 1, 0]])
x2 = torch.tensor([[1, 3], [0, 2], [0, 2]], dtype=torch.int32)
x3 = torch.tensor([[0, 1], [1, 0]])
W = torch.randint(low=0, high=5, size=(4, 3), generator=g)
# out.extend(torch.tensor() for row in x2)
out = (W[x2].sum(dim=1)).tolist()
print(W)
print(x1@W)
# print(x2)
print(torch.tensor(out))
