import argparse
import copy
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from sklearn import datasets as ds

import flows as fnn
from data import MSRADataset, get_hand
import geometry as geo


if __name__ == '__main__':
  dim_in = num_inputs = 2
  num_hidden = 64
  lr = 0.0001
  log_interval = 1000
  num_blocks = 5
  epochs = 5
  batch_size = 100
  
  """param.(batch_size, lr, total_it)"""
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

  #ds = MSRADataset(image=False, rotate=False)
  x = ds.make_moons(n_samples=100000, shuffle=True, noise=0.05)[0].astype(np.float32)
  class Moon(Dataset):
      def __init__(self, x):
          self.x = x.astype(np.float32)
      def __len__(self):
          return self.x.shape[0]
      def __getitem__(self, idx):
          return torch.from_numpy(self.x[idx])

  train_loader = DataLoader(
    Moon(x),
    num_workers=4,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True)
  
  modules = []
  
  for i_block in range(num_blocks):
    if i_block == num_blocks - 1:
        modules += [
            fnn.MADE(num_inputs, num_hidden),
            fnn.Reverse(num_inputs)]
    else:
        modules += [
            fnn.MADE(num_inputs, num_hidden),
            #fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)]
  
  model = fnn.FlowSequential(*modules)
  
  for module in model.modules():
    if isinstance(module, nn.Linear):
      nn.init.orthogonal_(module.weight)
      module.bias.data.fill_(0)
  
  model.to(device)
  
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
  
  
  def flow_loss(u, log_jacob, size_average=True):
    log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
        -1, keepdim=True)
    loss = -(log_probs + log_jacob).sum()
    if size_average:
      loss /= u.size(0)
    return loss
  
  
  def train(epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
      if isinstance(data, list):
        data = data[0]
      data = data.to(device)
      optimizer.zero_grad()
      u, log_jacob = model(data)
      loss = flow_loss(u, log_jacob)
      loss.backward()
      optimizer.step()
 
    print('Epoch {:d} of {:d}:\tLoss: {:.6f}'.format(epoch, epochs, loss.item()))
  
    for module in model.modules():
      if isinstance(module, fnn.BatchNormFlow):
        module.momentum = 0
  
    with torch.no_grad():
      model(train_loader.dataset[0].view(1,-1).to(data.device))
  
    for module in model.modules():
      if isinstance(module, fnn.BatchNormFlow):
        module.momentum = 1

  for epoch in range(epochs):
    train(epoch)

  torch.save(model.state_dict(), 'models/maf_model.pytorch')

  model.eval()
  with torch.no_grad():
    #subject, seq, idx = 'P0', '5', 0
    #img, pose = get_hand(subject, seq, idx=idx)
    z = np.random.randn(200, num_inputs).astype(np.float32)
    z_tens = torch.from_numpy(z).to(device)
    synth = model.forward(z_tens, mode='inverse', logdets=None)[0].detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x[:,0], x[:,1], '.')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(synth[:,0], synth[:,1], '.')
    # for i in range(4):
    #   ax = fig.add_subplot('22{}'.format(i+1), projection='3d')
    #   geo.plot_skeleton3d(synth[i], ax, autoscale=False)
    plt.show()
