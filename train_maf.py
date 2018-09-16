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

import flows as fnn
from data import MSRADataset, Moon, get_hand
import geometry as geo


if __name__ == '__main__':
  dataset = 'hands'
  num_hidden = 256
  lr = 0.0001
  log_interval = 1000
  num_blocks = 7
  epochs = 30
  batch_size = 100
  gestures = None
  
  
  """param.(batch_size, lr, total_it)"""
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

  if dataset == 'hands':
    ds = MSRADataset(image=False, angles=True, gestures=gestures, rotate=True)

  elif dataset == 'moons':
    ds = Moon()

  else:
    raise ValueError('no such dataset')

  dim_in = num_inputs = ds.n_dim
  train_loader = DataLoader(
    ds,
    num_workers=4,
    batch_size=batch_size,
    shuffle=True)
  
  model = fnn.FlowSequential(num_blocks, num_inputs, num_hidden, device)
  
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
      optimizer.zero_grad()
      jts, azim, elev = data['jts'].to(device), data['azim'].to(device), data['elev'].to(device)

      loss = -model.log_prob(jts, azim, elev).mean()
      loss.backward(retain_graph=True)
      optimizer.step()
 
    print('Epoch {:d} of {:d}:\tLoss: {:.6f}'.format(epoch, epochs, loss.item()))
  
    for module in model.modules():
      if isinstance(module, fnn.BatchNormFlow):
        module.momentum = 0
  
    #with torch.no_grad():
    #  model(train_loader.dataset[0].view(1,-1).to(data.device))
  
    for module in model.modules():
      if isinstance(module, fnn.BatchNormFlow):
        module.momentum = 1

  for epoch in range(epochs):
    train(epoch)

  torch.save(model.state_dict(), 'models/maf_model.pytorch')

  model.eval()
  with torch.no_grad():
    n_gen = 9 if dataset == 'hands' else 500
    z = np.random.randn(n_gen, num_inputs).astype(np.float32)
    z_tens = torch.from_numpy(z).to(device)
    synth = model.forward(z_tens, mode='inverse', logdets=None)[0].detach().cpu().numpy()

    fig = plt.figure()

    if dataset == 'hands':
      subject, seq, idx = 'P0', '5', 0
      img, pose = get_hand(subject, seq, idx=idx)
      for i in range(n_gen):
         ax = fig.add_subplot('33{}'.format(i+1), projection='3d')
         geo.plot_skeleton3d(synth[i], ax, autoscale=False)
    
    elif dataset == 'moons':
      ax = fig.add_subplot(111)
      ax.plot(ds.x[:,0], ds.x[:,1], '.')

      fig = plt.figure()
      ax = fig.add_subplot(111)
      ax.plot(synth[:,0], synth[:,1], '.')

    plt.show()
