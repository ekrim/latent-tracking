import sys
import os
import argparse
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

from data import MSRADataset
import geometry as geo
from model import RealNVP


PRINT_FREQ = 2000  # print loss every __ samples seen


def train(param, dim_in=63):
  """param.(batch_size, lr, total_it)"""
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

  ds = MSRADataset(image=False, rotate=param.rotate)
  dl = DataLoader(
    ds,
    num_workers=4,
    batch_size=param.batch_size,
    shuffle=True)
  
  flow = RealNVP(dim_in, device, grouped_mask=param.grouped_mask)
  flow.to(device)
  flow.train()

  optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad == True], lr=param.lr)

  it, print_cnt = 0, 0
  while it < param.total_it:
    for i, data in enumerate(dl):
      loss = -flow.log_prob(data.to(device)).mean()

      optimizer.zero_grad()
      loss.backward(retain_graph=True)
      optimizer.step()
   
      it += data.shape[0]
      print_cnt += data.shape[0]
      if print_cnt > PRINT_FREQ:
        print('it {:d} -- loss {:.03f}'.format(it, loss))
        print_cnt = 0

    torch.save(flow.state_dict(), 'models/flow_model.pytorch')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', default=64, type=int, help='batch size')
  parser.add_argument('--total_it', default=10000, type=int, help='number of training samples')
  parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
  parser.add_argument('--grouped_mask', action='store_true', help='group masks into chunks of xyz pts')
  parser.add_argument('--rotate', action='store_true', help='add y-axis rotated data to train set')

  args = parser.parse_args(sys.argv[1:])
  train(args)
