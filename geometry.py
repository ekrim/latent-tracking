import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  x_orig = np.array(
    [[0, 0, 1],
     [0.5, 0.5, 0.2]])

  x_new = rotate(x_orig, np.pi/2, axis='y') 
  
  ax.plot(x_orig[:,0], x_orig[:,1], x_orig[:,2], '.')
  ax.plot(x_new[:,0], x_new[:,1], x_new[:,2], '.r')
  plt.show()
  

