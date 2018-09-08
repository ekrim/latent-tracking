import numpy as np
import torch
import torch.nn as nn


class RealNVP(nn.Module):
  def __init__(self, dim_in, device):
    super(RealNVP, self).__init__()

    n_hid = 256
    n_mask = 10
    
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
