import os
import sys
import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot3d(x, col='b'):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(x[:,0], x[:,1], x[:,2], '.'+col)
  ax.set_xlabel('y')
  ax.set_ylabel('x')
  ax.set_zlabel('z')
  return ax


def plot_skeleton(x):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  fingers = [1,2,3,4]
  ax.plot(x[fingers, 0], x[fingers, 1], x[fingers, 2], '.')
  ax.plot([x[0, 0]], [x[0, 1]], [x[0, 2]], 'g*')
  ax.plot([x[5, 0]], [x[5, 1]], [x[5, 2]], 'rs')
  for i in range(5):
    ax.plot(x[[i,5], 0], x[[i,5], 1], x[[i,5], 2], 'b')
  ax.set_xlabel('y')
  ax.set_ylabel('x')
  ax.set_zlabel('z')
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
  plot3d(joints)

  plt.show()


