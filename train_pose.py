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

from data import MRSADataset
from model import PoseModel


PRINT_FREQ = 2000  # print loss every __ samples seen


def train(param):
  """param.(batch_size, lr, total_it)"""
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

  ds = MRSADataset(image=True)
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

  it, print_cnt = 0, 0
  while it < param.total_it:
    for i, data in enumerate(dl):
      optimizer.zero_grad()
      inputs, labels = data['img'].to(device), data['jts'].to(device)
      outputs = pose_mod(inputs)   
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
   
      it += inputs.shape[0]
      print_cnt += inputs.shape[0]
      if print_cnt > PRINT_FREQ:
        print('it {:d} -- loss {:.03f}'.format(it, loss))
        print_cnt = 0

    torch.save(pose_mod.state_dict(), 'pose_model.pytorch')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', default=64, type=int, help='batch size')
  parser.add_argument('--total_it', default=10000, type=int, help='number of training samples')
  parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')

  args = parser.parse_args(sys.argv[1:])
  train(args)
