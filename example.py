

# %%
import torch 



x = [ torch.rand((32,18)) for _ in range(5)]

x = torch.stack(x, dim = 0)
mean_v = torch.mean(x, dim=0)

print(mean_v.shape)