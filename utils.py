import os
import sys
from glob import glob
import numpy as np


def all_joint_files():
  seq_dir_list = glob('annotated_frames/data_*')
  joint_files = [] 
  for seq_dir in seq_dir_list:
    joint_files += glob(os.path.join(seq_dir, '*_joints.txt'))
  return joint_files

def stack(lst):
  lst = [jts.reshape((1,-1)) for jts in lst]
  return np.concatenate(lst, axis=0)

def interpolate(x, n):
  out = np.zeros((n, x.shape[1])).astype(np.float32)
  wts = np.linspace(0, 1, n)
  for i, wt in enumerate(wts):
    out[i] = x[0] + wt*(x[1] - x[0])
  return out

def img_and_joint_files(i_seq, i_frame):
  base = 'annotated_frames/data_{:d}'.format(i_seq)
  fn_joints = os.path.join(base, '{:d}_joints.txt'.format(i_frame))
  fn_img = os.path.join(base, '{:d}_webcam_1.jpg'.format(i_frame))
  return fn_joints, fn_img


