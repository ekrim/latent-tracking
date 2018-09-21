import sys
import os
import argparse
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import flows
import geometry as geo
from data import MSRADataset, SUBJECTS
from model import PoseModel


PRINT_FREQ = 2000  # print loss every __ samples seen


def train(param):
  """param.(batch_size, lr, total_it)"""
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

  ds = MSRADataset(image=True, rotate=param.rotate, subjects=[s for s in SUBJECTS if s != 'P8'])
  dl = DataLoader(
    ds,
    num_workers=4,
    batch_size=param.batch_size,
    shuffle=True)
  
  pose_mod = PoseModel() 
  pose_mod.to(device)
  pose_mod.train()
  
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam([p for p in pose_mod.parameters() if p.requires_grad == True], lr=param.lr)

  # flow model
  if param.predict_z:
    flow_mod = flows.FlowSequential(param.flow_blocks, 63, param.flow_dim, device)
    flow_mod = geo.load_model(flow_mod, param.flow_model, device)
    flow_mod.eval()
    enc_fnc = lambda x: flow_mod.forward(torch.from_numpy(x.astype(np.float32)).to(device))[0]

  it, print_cnt = 0, 0
  while it < param.total_it:
    for i, data in enumerate(dl):
      optimizer.zero_grad()
      inputs, labels = data['img'].to(device), data['jts'].to(device)
      outputs = pose_mod(inputs)   
      if param.predict_z:
        with torch.no_grad():
          z_labels = flow_mod.forward(labels)[0]
        loss = criterion(outputs, z_labels)
      else:
        loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
   
      it += inputs.shape[0]
      print_cnt += inputs.shape[0]
      if print_cnt > PRINT_FREQ:
        print('it {:d} -- loss {:.03f}'.format(it, loss))
        print_cnt = 0

    torch.save(pose_mod.state_dict(), param.model)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', default=64, type=int, help='batch size')
  parser.add_argument('--total_it', default=300000, type=int, help='number of training samples')
  parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
  parser.add_argument('--model', default='models/pose_model.pytorch', help='file to save pose model to')
  parser.add_argument('--rotate', action='store_true', help='rotate training data')

  parser.add_argument('--predict_z', action='store_true', help='predict poses in z space')
  parser.add_argument('--flow_model', default='models/maf_all_10_256_30e.pytorch', help='trained flow model')
  parser.add_argument('--flow_blocks', default=10, type=int, help='number of stacked blocks of flow')
  parser.add_argument('--flow_dim', default=256, type=int, help='number of neurons in flow blocks')
  args = parser.parse_args(sys.argv[1:])
  train(args)
