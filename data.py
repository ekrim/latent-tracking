import sys
import os
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader


SUBJECTS = ['P{:d}'.format(i) for i in range(9)]
GESTURES = '1 2 3 4 5 6 7 8 9 I IP L MP RP T TIP Y'.split(' ')


def load_image_bin(f_name):
  f = open(f_name, 'rb')
  header = f.read(6*4)
  header = struct.unpack('6I', header)
  n_rows = header[5] - header[3]
  n_cols = header[4] - header[2]
  
  body = f.read(n_rows*n_cols*4)
  body = struct.unpack(str(n_rows*n_cols)+'f', body)
  img = np.array(body).reshape((n_rows, n_cols)).astype(np.float32)/255
  return img


def all_depth_and_joints(subjects, gestures):
  depth_files = [] 
  jts_list = [] 
  for subject in subjects:
    for gesture in gestures:
      joint_file = os.path.join('MRSA', subject, gesture, 'joint.txt')
      jts = load_joints(joint_file)
      jts_list += [jts]
      depth_files += [os.path.join('MRSA', subject, gesture, '{:06d}_depth.bin'.format(i)) for i in range(jts.shape[0])]
  return depth_files, np.concatenate(jts_list, axis=0) 


def load_joints(f):
  jts = pd.read_csv(f, sep=' ', header=None, skiprows=[0])
  return np.float32(jts)


class MRSADataset(Dataset):
  def __init__(self, subjects=None, gestures=None, image=False, max_buffer = 20):
    subjects = SUBJECTS if subjects is None else subjects
    gestures = GESTURES if gestures is None else gestures

    self.image = image
    self.depth_files, self.joints = all_depth_and_joints(subjects, gestures)

  def __len__(self):
    return len(self.depth_files)
    
  def __getitem__(self, idx): 
    jts_tens = torch.from_numpy(self.joints[idx])
    if self.image:
      img = load_image_bin(self.depth_files[idx])
      bigger_dim = np.max(img.shape) + np.random.randint(0, self.max_buffer) #TODO
      
      img_tens = torch.from_numpy(img)
      return {'img': img_tens, 'jts': jts_tens}
    else:
      return jts_tens



if __name__ == '__main__':
  ds = MRSADataset(image=True) 
  dl = DataLoader(
    ds,
    num_workers=4,
    batch_size=2,
    shuffle=False)

  for d in dl:
    print(d)
