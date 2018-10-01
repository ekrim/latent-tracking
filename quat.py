import numpy as np
import matplotlib.pyplot as plt
import geometry as geo


if __name__ == '__main__':
  x = np.float32([0, 1, 0])
 
  N = 1000
  x_new = np.zeros((N, 3))
  for i in range(N):
    q = geo.random_quaternion()
    x_new[i,:] = geo.quaternion_rotation(q, x)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  geo.plot3d(x[None,:], ax)
  geo.plot3d(x_new, ax, col='r')

  plt.show()

