import os
import sys
import PIL
from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

import data


JOINTS = 'wrist index_mcp index_pip index_dip index_tip middle_mcp middle_pip middle_dip middle_tip ring_mcp ring_pip ring_dip ring_tip little_mcp little_pip little_dip little_tip thumb_mcp thumb_pip thumb_dip thumb_tip'.split(' ')

MS = 20
MS2 = 30

HandPart = namedtuple('HandPart', 'connections conn_color pts pt_color pt_size')

INDEX = HandPart(
  connections=np.array([[4,3], [3,2], [2,1]]),
  conn_color='g',
  pts=[1,2,3,4],
  pt_color='g',
  pt_size=MS)

MIDDLE = HandPart(
  connections=np.array([[8,7], [7,6], [6,5]]),
  conn_color='m',
  pts=[5,6,7,8],
  pt_color='m',
  pt_size=MS)

RING = HandPart(
  connections=np.array([[12,11], [11,10], [10,9]]),
  conn_color='r',
  pts=[9,10,11,12],
  pt_color='r',
  pt_size=MS)

LITTLE = HandPart(
  connections=np.array([[16,15], [15,14], [14,13]]),
  conn_color='c',
  pts=[13,14,15,16],
  pt_color='c',
  pt_size=MS)

THUMB = HandPart(
  connections=np.array([[20,19], [19,18], [18,17]]),
  conn_color='y',
  pts=[17,18,19,20],
  pt_color='y',
  pt_size=MS)

WRIST = HandPart(
  connections=np.array([[0,17], [0,1], [0,5], [0,9], [0,13]]),
  conn_color='b',
  pts=[0],
  pt_color='b',
  pt_size=MS2)

HAND = [INDEX, MIDDLE, RING, LITTLE, THUMB, WRIST]


def plot3d(x, ax, col='b', ms=10):
  ax.plot(x[:,0], x[:,1], x[:,2], '.'+col, markersize=ms)
  ax.set_xlabel('y')
  ax.set_ylabel('x')
  ax.set_zlabel('z')
  return ax


def plot_skeleton(x, ax):
  if x.shape[-1] != 3:
    x = x.reshape((-1, 3))

  for obj in HAND:
    ax = plot3d(x[obj.pts], ax, col=obj.pt_color, ms=obj.pt_size)
    for idx_pair in obj.connections:
      ax.plot(x[idx_pair, 0], x[idx_pair, 1], x[idx_pair, 2], obj.conn_color)

  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.set_aspect('equal')

  ranges = np.concatenate([ 
    np.min(x, axis=0)[None,:],
    np.max(x, axis=0)[None,:]], axis=0)

  max_range = np.ceil(np.max(ranges[1] - ranges[0]))
  mean_range = np.mean(ranges, axis=0)
 
  new_range = np.concatenate([
    (mean_range-max_range/2)[None,:],
    (mean_range+max_range/2)[None,:],
  ])
  
  ax.auto_scale_xyz(new_range[:,0], new_range[:,1], new_range[:,2])
  return ax


def plot_skeleton2d(x, ax, autoscale=True):
  if x.shape[-1] != 3:
    x = x.reshape((-1, 3))

  for obj in HAND:
    ax.plot(x[obj.pts, 0], x[obj.pts, 1], '.'+obj.pt_color, markersize=obj.pt_size)
    for idx_pair in obj.connections:
      ax.plot(x[idx_pair, 0], x[idx_pair, 1], obj.conn_color)

  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_aspect('equal')

  if autoscale:
    ranges = np.concatenate([ 
      np.min(x, axis=0)[None,:],
      np.max(x, axis=0)[None,:]], axis=0)

    max_range = np.ceil(np.max(ranges[1] - ranges[0]))
    mean_range = np.mean(ranges, axis=0)
   
    new_range = np.concatenate([
      (mean_range-max_range/2)[None,:],
      (mean_range+max_range/2)[None,:],
    ])
    
    ax.set_xlim(new_range[:,0])  
    ax.set_ylim(new_range[:,1])  
  return ax


def joints_over_depth(jts, img, ax):
  #ax.imshow(img.transpose(PIL.Image.FLIP_TOP_BOTTOM))
  ax.imshow(img[::-1,:], cmap='Greys_r')

  n_rows, n_cols = img.shape
  
  bounds = find_bounds(img)

  new_jts = jts.copy().reshape((-1,3))

  new_jts = normalize_dim(new_jts, bounds[3], bounds[1], 1)
  x_min = np.min(new_jts[:,0])
  new_jts[:,0] += bounds[0] - x_min
  
  plot_skeleton2d(new_jts, ax, autoscale=False)
  return ax


def find_bounds(img, thresh=0.7):
  truth = img > thresh  
  col_bounds = np.arange(img.shape[1])[np.any(truth, axis=0)][[0,-1]]
  row_bounds = np.arange(img.shape[0])[np.any(truth, axis=1)][[0,-1]]
  return col_bounds[0], img.shape[0]-row_bounds[0], col_bounds[1], img.shape[0]-row_bounds[1]


def normalize_dim(x, goal_min, goal_max, dim):
  # put in [0,1]
  min_val = np.min(x[:,dim])
  x -= min_val
  max_val = np.max(x[:,dim])
  x = x/max_val

  mag = goal_max - goal_min
  x *= mag
  x[:,dim] += goal_min
  
  return x
  
 
def rotate(x, theta, axis='x'):
  if axis == 'x':
    R = np.float32(
      [[1, 0, 0], 
       [0, np.cos(theta), -np.sin(theta)],
       [0, np.sin(theta), np.cos(theta)]])
  elif axis == 'y':
    R = np.float32(
      [[np.cos(theta), 0, np.sin(theta)],
       [0, 1, 0],
       [-np.sin(theta), 0, np.cos(theta)]])
  elif axis == 'z':
    R = np.float32(
      [[np.cos(theta), -np.sin(theta), 0],
       [np.sin(theta), np.cos(theta), 0],
       [0, 0, 1]])

  return np.dot(R, x.transpose()).transpose()


def load_model(mod, mod_file, device):
  if device.type == 'cpu':
    mod.load_state_dict(torch.load(mod_file, map_location='cpu'))
  else:
    mod.load_state_dict(torch.load(mod_file))

  mod.to(device)
  mod.eval()
  return mod


def stack(lst):
  lst = [jts.reshape((1,-1)) for jts in lst]
  return np.concatenate(lst, axis=0)


def interpolate(x, n):
  out = np.zeros((n, x.shape[1])).astype(np.float32)
  wts = np.linspace(0, 1, n)
  for i, wt in enumerate(wts):
    out[i] = x[0] + wt*(x[1] - x[0])
  return out


if __name__ == '__main__':
  subject = sys.argv[1]
  sequence = sys.argv[2]
  
  joint_file = 'MSRA/{}/{}/joint.txt'.format(subject, sequence)
  print(joint_file)
  jts = data.load_joints(joint_file)
  
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  plot_skeleton(jts[0], ax)

  img = PIL.Image.open('MSRA/P4/9/000000_depth.jpg')

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax = joints_over_depth(jts, img, ax)
  plt.show()
  
