import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


def kalman_filter3d(z, P=0.5, Q=0.5, R=10.0, dt=0.5):
  """
  Args:
    z: (n_time_steps, 3*n_pts) ndarray of pts to be tracked
    P: error covariance
    Q: process noise
    R: measurement noise
    dt: time step delta

  """
  n_dim = z.shape[1]

  # one filter per joint 
  filters = [KalmanFilter(dim_x=6, dim_z=3) for i in range(n_dim//3)]
  for i, f in enumerate(filters):
    f.x = np.append(z[0, 3*i:3*(i+1)], [0, 0, 0])

    f.F = np.float32([
      [1, 0, 0, dt,  0,  0],
      [0, 1, 0,  0, dt,  0],
      [0, 0, 1,  0,  0, dt],
      [0, 0, 0,  1,  0,  0],
      [0, 0, 0,  0,  1,  0],
      [0, 0, 0,  0,  0,  1]])

    f.H = np.float32([
      [1, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0]])

    f.P *= P
    f.R *= R
    f.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q, block_size=3, order_by_dim=False)

  z_smooth = np.zeros(z.shape)
  for i_row in range(z.shape[0]):
    for i_pt, f in enumerate(filters):
      f.predict()
      f.update(z[i_row, 3*i_pt:3*(i_pt+1)])
      z_smooth[i_row, 3*i_pt:3*(i_pt+1)] = f.x[:3].copy()

  return z_smooth


def example_path():
  return np.array([
    [59,100 ],
    [73,80  ], 
    [101,96 ], 
    [131,77 ], 
    [146,98 ], 
    [130,129], 
    [157,153],
    [193,103],
    [254,80 ],
    [269,114],
    [310,109],
    [366,83 ], 
    [372,138],
    [326,150],
    [284,183],
    [304,215],
    [343,198],
    [370,221],
    [325,233],
    [289,266],
    [369,270],
    [323,278],
    [260,311],
    [339,311],
    [300,315],
    [284,361],
    [363,357],
    [389,317],
    [438,357],
    [472,419],
    [483,359],
    [520,385],
    [553,446],
    [565,377],
    [606,435]])/610


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dt', default=0.5, type=float, help='time delta')
  parser.add_argument('--P', default=0.13, type=float, help='covariance matrix')
  parser.add_argument('--R', default=0.5, type=float, help='measurement noise')
  parser.add_argument('--Q', default=0.5, type=float, help='process noise')
  args = parser.parse_args(sys.argv[1:])
  
  pts = example_path()

  dt = args.dt
  f = KalmanFilter(dim_x=4, dim_z=2)
  f.x = np.append(pts[0], [0, 0])
  f.F = np.float32([
    [1,0,dt,0],
    [0,1,0,dt],
    [0,0,1,0],
    [0,0,0,1]])
    
  f.H = np.float32([
    [1,0,0,0],
    [0,1,0,0]])

  f.P *= args.P
  f.R *= args.R
  
  f.Q = Q_discrete_white_noise(dim=2, dt=dt, var=args.Q, block_size=2, order_by_dim=False)

  #mu, cov, _, _ = f.batch_filter(pts)
  #new_pts, P, C, _ = f.rts_smoother(mu, cov)
  
  new_pts = np.zeros(pts.shape)
  for i, pt in enumerate(pts):
    f.predict()
    f.update(pt)
    new_pts[i] = f.x[:2].copy()
    
  plt.plot(pts[:,0], pts[:,1], '.-')
  plt.plot(new_pts[:,0], new_pts[:,1], '.-r')

  plt.show()
