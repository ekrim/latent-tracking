import sys
import numpy as np
import torch
import torch.nn as nn


class Mod(nn.Sequential):
  def __init__(self):
    super().__init__()

  def create(self, m, n):
    self.fuck = nn.Parameter(torch.arange(10), requires_grad=False)

    m = torch.tensor(m)
    n = torch.tensor(n)

    self.register_buffer('m', m)
    self.register_buffer('n', n)
    self.main = nn.Sequential(
      nn.Linear(m.numpy(), n.numpy()))

    mean = torch.zeros(2)
    std = torch.eye(2)
    self.prior = torch.distributions.MultivariateNormal(mean, std)
    self.register_buffer('mean', mean)
    self.register_buffer('std', std)
    return self

  def load(self, f):
    state_dict = torch.load(f)
    m = state_dict['m'].numpy()
    n = state_dict['n'].numpy()

    self.create(m, n)
    self.load_state_dict(state_dict)

  def forward(self, x):
    return x

if __name__ == '__main__':
  mode = sys.argv[1]
  if mode == 'save':
    mod = Mod().create(3, 5)

    print(mod.state_dict())
    torch.save(mod.state_dict(), 'fuck.pytorch')
  elif mode == 'load':
    mod = Mod()
    mod.load('fuck.pytorch')
    print(mod.state_dict())
    print(mod.prior)
    print(mod.fuck)

