import torch 

a = torch.Tensor([0.02, 0.9, 0.05, 0.03]) 
dist = torch.distributions.Categorical(a) 
print(dist.sample().item())