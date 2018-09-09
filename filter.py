import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


PTS = np.array([
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
  parser.add_argument('--P', default=0.5, type=float, help='covariance matrix')
  parser.add_argument('--R', default=0.5, type=float, help='measurement noise')
  parser.add_argument('--Q', default=0.13, type=float, help='process noise')
  args = parser.parse_args(sys.argv[1:])

  dt = args.dt
  f = KalmanFilter(dim_x=4, dim_z=2)
  f.x = np.append(PTS[0], [0, 0])
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
  
  f.Q = Q_discrete_white_noise(dim=4, dt=dt, var=args.Q)
  
  new_pts = np.zeros(PTS.shape)
  for i, pt in enumerate(PTS):
    f.predict()
    f.update(pt)
    new_pts[i] = f.x[:2].copy()
    
  plt.plot(PTS[:,0], PTS[:,1], '.-')
  plt.plot(new_pts[:,0], new_pts[:,1], '.-r')

  plt.show()
