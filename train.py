import sys
import os
import argparse
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

from geometry import plot_skeleton
from model import RealNVP


PRINT_FREQ = 2000  # print loss every __ samples seen


def train(param, dim_in=63):
  """param.(batch_size, lr, total_it)"""
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  print(device)

  ds = HandDataset()
  dl = DataLoader(
    ds,
    num_workers=4,
    batch_size=param.batch_size,
    shuffle=True)
  
  flow = RealNVP(dim_in, device) 
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

    torch.save(flow.state_dict(), 'flow_model.pytorch')


class HandDataset(Dataset):
  def __init__(self):
    seq_dir_list = glob('annotated_frames/data_*')
    self.joint_files = [] 
    for seq_dir in seq_dir_list:
      self.joint_files += glob(os.path.join(seq_dir, '*_joints.txt'))
  
  def __len__(self):
    return len(self.joint_files)
    
  def __getitem__(self, idx): 
    joints = pd.read_csv(self.joint_files[idx], sep=' ', header=None)
    joints = np.float32(joints.iloc[:-1, 1:]).flatten()
    joints = (joints - 33.55)/108.88
    return torch.from_numpy(joints)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', default=64, type=int, help='batch size')
  parser.add_argument('--total_it', default=10000, type=int, help='number of training samples')
  parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')

  args = parser.parse_args(sys.argv[1:])
  train(args)
