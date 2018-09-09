"""TODO:
- Find finger movement example
"""
import sys
import os
import argparse
import PIL
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import utils
import geometry as geo
from model import RealNVP


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', default='flow_model.pytorch', help='trained RealNVP model')
  parser.add_argument('--dim_in', default=63, type=int, help='dimensionality of input data')
  parser.add_argument('--display', action='store_true', help='generate samples and display them')
  parser.add_argument('--interpolate', action='store_true', help='run interpolation experiment')
  parser.add_argument('--rotation', action='store_true', help='run rotation semantiic direction experiment')
  parser.add_argument('--inverse_kinematics', action='store_true', help='run inverse kinematics experiment')
  args = parser.parse_args(sys.argv[1:])

  joint_files = utils.all_joint_files()    

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
  
  
  if args.display:
    # display generated
    hand_pose = gen_fnc(np.random.randn(9, 63))
    
    fig = plt.figure()
    for i in range(9):
      ax = fig.add_subplot('33{:d}'.format(i+1), projection='3d')
      geo.plot_skeleton(hand_pose[i], ax, col='b')


  if args.interpolate:   
    # interpolation in latent space
    joint_f1, img_f1 = utils.img_and_joint_files(5, 33)
    joint_f2, img_f2 = utils.img_and_joint_files(10, 522)

    pose1 = geo.load_joints(joint_f1)
    pose2 = geo.load_joints(joint_f2)

    # interp in latent space
    z = enc_fnc(utils.stack([pose1, pose2]))
    z_interp = utils.interpolate(z, 9) 
    pose_interp = gen_fnc(z_interp)

    fig = plt.figure()
    for i in range(9):
      ax = fig.add_subplot('33{:d}'.format(i+1), projection='3d')
      geo.plot_skeleton(pose_interp[i], ax, col='r')

    # interp in data space
    x_interp = utils.interpolate(utils.stack([pose1, pose2]), 9)

    fig = plt.figure()
    for i in range(9):
      ax = fig.add_subplot('33{:d}'.format(i+1), projection='3d')
      geo.plot_skeleton(x_interp[i], ax, col='r')
 
    # show images
    fig = plt.figure()
    ax = fig.add_subplot(211)
    img = PIL.Image.open(img_f1)
    ax.imshow(img)
    ax = fig.add_subplot(212)
    img = PIL.Image.open(img_f2)
    ax.imshow(img)
  
  
  if args.rotation:
    # find direction of rotation in latent space
    n_samples = 1000
    joint_file_sample = np.random.choice(joint_files, n_samples, replace=False)
    original = [geo.load_joints(f)[None,:] for f in joint_file_sample]
    x_original = np.concatenate(original, axis=0)
    z_original = enc_fnc(x_original)
    
    rotation_vecs = []
    for axis in ['x', 'y', 'z']:
      for sign in [-1,1]:
        theta_vec = sign * np.pi/6 * np.ones(n_samples) #sign * np.pi/2 * np.random.rand(n_samples)
        rotated = [geo.rotate(jts.reshape((-1, 3)), theta, axis=axis).flatten()[None,:] for jts, theta in zip(original, theta_vec)]
        x_rotated = np.concatenate(rotated, axis=0)
        z_rotated = enc_fnc(x_rotated) 
        
        rotation_vecs += [(z_original, z_rotated)]
      
    # test and plot  
    joint_file = np.random.choice(joint_files)
    test_joint = geo.load_joints(joint_file)[None,:]
    z = enc_fnc(test_joint)
    fig = plt.figure()
    ax = fig.add_subplot(241, projection='3d')
    geo.plot_skeleton(test_joint, ax, col='r')
    for i in range(3):
      for row in range(2):
        closest_idx = np.argmin(np.sum((z - rotation_vecs[2*i + row][0])**2, axis=1))
        z_rot = z + (rotation_vecs[2*i + row][1][closest_idx][None,:] - z)

        x_rot = gen_fnc(z_rot)
        ax = fig.add_subplot('24{:d}'.format(i+2+row*4), projection='3d')
        geo.plot_skeleton(x_rot, ax, col='r')


  if args.inverse_kinematics:
    # run inverse kinematics experiments
    f = np.random.choice(joint_files)
    jts = geo.load_joints(f).reshape((-1, 3))
    bad_jts = jts.copy()
    bad_jts[7] = jts[3].copy()
    bad_jts[3] = jts[7].copy()
 
    z_bad = enc_fnc(bad_jts.reshape((1, -1)))
    z_bad *= 0.7
    x_bad = gen_fnc(z_bad)
   
    fig = plt.figure()
    ax = fig.add_subplot(131, projection='3d')
    ax = geo.plot_skeleton(bad_jts, ax, col='r')
    ax = fig.add_subplot(132, projection='3d')
    ax = geo.plot_skeleton(x_bad, ax, col='r')
    ax = fig.add_subplot(133, projection='3d')
    ax = geo.plot_skeleton(jts, ax, col='r')
    

  plt.show() 
