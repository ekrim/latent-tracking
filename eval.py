import sys
import os
import argparse
import PIL
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import geometry as geo
from model import RealNVP



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


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', default='flow_model.pytorch', help='trained RealNVP model')
  parser.add_argument('--dim_in', default=63, type=int, help='dimensionality of input data')
  args = parser.parse_args(sys.argv[1:])


  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  flow = RealNVP(args.dim_in, device)
  if device.type == 'cpu':
    flow.load_state_dict(torch.load(args.model, map_location='cpu'))
  else:
    flow.load_state_dict(torch.load(args.model))

  flow.to(device)
  flow.eval()

  gen_fnc = lambda z: flow.g(torch.from_numpy(z.astype(np.float32)).to(device)).detach().cpu().numpy()
  enc_fnc = lambda x: flow.f(torch.from_numpy(x.astype(np.float32)).to(device))[0].detach().cpu().numpy()
  
  
  # display generated
  hand_pose = gen_fnc(np.random.randn(9, 63))
  
  fig = plt.figure()
  for i in range(9):
    ax = fig.add_subplot('33{:d}'.format(i+1), projection='3d')
    geo.plot_skeleton(hand_pose[i], ax, col='b')

    
  # interpolation in latent space
  joint_f1, img_f1 = img_and_joint_files(1, 10)
  joint_f2, img_f2 = img_and_joint_files(9, 100)

  pose1 = geo.load_joints(joint_f1)
  pose2 = geo.load_joints(joint_f2)
  z = enc_fnc(stack([pose1, pose2]))
 
  z_interp = interpolate(z, 9) 
  pose_interp = gen_fnc(z_interp)

  fig = plt.figure()
  for i in range(9):
    ax = fig.add_subplot('33{:d}'.format(i+1), projection='3d')
    geo.plot_skeleton(pose_interp[i], ax, col='b')

  fig = plt.figure()
  ax = fig.add_subplot(211)
  img = PIL.Image.open(img_f1)
  ax.imshow(img)
  ax = fig.add_subplot(212)
  img = PIL.Image.open(img_f2)
  ax.imshow(img)
  


  plt.show() 
