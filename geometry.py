import os
import sys
import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


HAND_CONNECT = np.array([
  # index
  [3, 2],
  [2, 0],
  [0, 1],
  [1, 5],

  [7, 6],
  [6, 4],
  [4, 5],
  [5, 13],

  [15, 14],
  [14, 12],
  [12, 13],
  [13, 9],

  [11, 10],
  [10, 8],
  [8, 9],

  # thumb
  [19, 18],
  [18, 16],
  [16, 17],
  [17, 1],
])


def plot3d(x, col='b'):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(x[:,0], x[:,1], x[:,2], '.'+col)
  ax.set_xlabel('y')
  ax.set_ylabel('x')
  ax.set_zlabel('z')
  return ax


def plot_skeleton(x):
  ax = plot3d(x)
  for idx_pair in HAND_CONNECT:
    ax.plot(x[idx_pair, 0], x[idx_pair, 1], x[idx_pair, 2], 'b')
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


def pts_over_depth(pts, depth):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.imshow(depth)

  masked = np.max(pts, axis=2) 
  masked[masked < 0.4] = 0.0

  rgba = np.concatenate([
    masked[:,:,None],
    np.zeros(depth.shape+(2,)),
    masked[:,:,None]], axis=2)

  ax.imshow(rgba)
  
  
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


if __name__ == '__main__':
  i_seq = int(sys.argv[1])
  i_frame = int(sys.argv[2])

  path = './annotated_frames/data_{:d}/'.format(i_seq)
  image_files = [os.path.join(path, '{:d}_webcam_{:d}.jpg'.format(i_frame, i_cam)) for i_cam in range(1,5)]
  joints_file = os.path.join(path, '{:d}_joints.txt'.format(i_frame)) 

  for fn in image_files:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    img = PIL.Image.open(fn)
    ax.imshow(img)

  print(joints_file)
  joints = pd.read_csv(joints_file, sep=' ', header=None)
  joints = np.array(joints.iloc[:,1:])
  plot_skeleton(joints)
  
  new_joints = rotate(joints, np.pi/2)
  plot_skeleton(new_joints)

  plt.show()


