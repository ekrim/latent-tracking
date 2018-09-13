import numpy as np
import torch
import torch.nn as nn


def ind_to_mask(n_dim, lst):
  if type(lst[0]) is list:
    mask = np.zeros((len(lst), n_dim))
    for i, sub_lst in enumerate(lst):
      idx = (3*np.array(sub_lst)[:,None] + np.arange(3)[None,:]).flatten()
      mask[i, idx] = 1 

  else:
    mask = np.zeros(n_dim)
    idx = (3*np.array(lst)[:,None] + np.arange(3)[None,:]).flatten()
    mask[idx] = 1

  return mask.astype(np.float32)
 

class RealNVP(nn.Module):
  def __init__(self, dim_in, device, n_hid=256, n_mask=10, grouped_mask=False):
    super(RealNVP, self).__init__()

    nets = lambda: nn.Sequential(
      nn.Linear(dim_in, n_hid), 
      nn.LeakyReLU(), 
      nn.BatchNorm1d(n_hid),
      nn.Linear(n_hid, n_hid), 
      nn.LeakyReLU(), 
      nn.BatchNorm1d(n_hid),
      nn.Linear(n_hid, dim_in), 
      nn.Tanh())

    nett = lambda: nn.Sequential(
      nn.Linear(dim_in, n_hid), 
      nn.LeakyReLU(), 
      nn.BatchNorm1d(n_hid),
      nn.Linear(n_hid, n_hid), 
      nn.LeakyReLU(), 
      nn.BatchNorm1d(n_hid),
      nn.Linear(n_hid, dim_in))

    if grouped_mask:
      mask_list = []
      for i_mask in range(n_mask):
        mask_list += [list(np.sort(np.random.choice(dim_in//3, int(np.ceil(dim_in/6)), replace=False)))]
      masks = torch.from_numpy(ind_to_mask(dim_in, mask_list))

    else:
      masks = torch.from_numpy(np.random.randint(0, 2, (n_mask, dim_in)).astype(np.float32))

    prior = torch.distributions.MultivariateNormal(torch.zeros(dim_in).to(device), torch.eye(dim_in).to(device))

    self.device = device
    self.prior = prior
    self.mask = nn.Parameter(masks, requires_grad=False)
    self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
    self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])
    
  def g(self, z):
    x = z
    for i in range(len(self.t)):
      x_ = x*self.mask[i]
      s = self.s[i](x_)*(1 - self.mask[i])
      t = self.t[i](x_)*(1 - self.mask[i])
      x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
    return x

  def f(self, x):
    log_det_J, z = x.new_zeros(x.shape[0]), x
    for i in reversed(range(len(self.t))):
      z_ = self.mask[i] * z
      s = self.s[i](z_) * (1-self.mask[i])
      t = self.t[i](z_) * (1-self.mask[i])
      z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
      log_det_J -= s.sum(dim=1)
    return z, log_det_J
  
  def log_prob(self, x):
    z, logp = self.f(x)
    return self.prior.log_prob(z) + logp
    
  def sample(self, batchSize): 
    z = self.prior.sample((batchSize, 1))
    logp = self.prior.log_prob(z)
    x = self.g(z)
    return x


class Flatten(nn.Module):
  def __init__(self, size):
    super().__init__()
    self.size = size
  
  def forward(self, x):
    return x.view(-1, self.size)


class PoseModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.AvgPool2d(2),
      nn.Conv2d(1, 16, 3),
      nn.LeakyReLU(), 
      nn.AvgPool2d(3),
      nn.BatchNorm2d(16),
      nn.Conv2d(16, 32, 3),
      nn.LeakyReLU(), 
      nn.AvgPool2d(2),
      nn.BatchNorm2d(32),
      Flatten(512),
      nn.Linear(512, 144),
      nn.ReLU(),
      nn.Linear(144, 144),
      nn.ReLU(),
      nn.Linear(144, 144),
      nn.ReLU(),
      nn.Linear(144, 144),
      nn.ReLU(),
      nn.Linear(144, 144),
      nn.ReLU(),
      nn.Linear(144, 63))
    
  def forward(self, x):
    for lay in self.layers:
      x = lay(x)
    return x


if __name__ == '__main__':
  mod = PoseModel()
  x = np.random.randn(4, 1, 64, 64).astype(np.float32)
  x = torch.from_numpy(x)
  print(x.shape)
  output = mod(x)
  print(output.detach().cpu().numpy())
  print(output.shape)
