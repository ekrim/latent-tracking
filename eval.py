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
from torch.utils.data import Dataset, DataLoader

import geometry as geo
import kf
from data import MSRADataset, SUBJECTS, GESTURES, get_hand
from model import RealNVP, PoseModel


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--flow_model', default='models/flow_model.pytorch', help='trained RealNVP model')
  parser.add_argument('--pose_model', default='models/pose_model.pytorch', help='trained pose model')
  parser.add_argument('--dim_in', default=63, type=int, help='dimensionality of input data')
  parser.add_argument('--display_pose', action='store_true', help='display pose estimates')
  parser.add_argument('--filter_pose', action='store_true', help='kalman filter pose smoothing')
  parser.add_argument('--display_flow', action='store_true', help='generate synthetic poses and display them')
  parser.add_argument('--interpolate', action='store_true', help='run interpolation experiment')
  parser.add_argument('--neighborhood', action='store_true', help='run neighborhood generation')
  parser.add_argument('--inverse_kinematics', action='store_true', help='run inverse kinematics experiment')
  args = parser.parse_args(sys.argv[1:])

  dim_in = args.dim_in
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  flow_mod = RealNVP(args.dim_in, device)
  flow_mod = geo.load_model(flow_mod, args.flow_model, device)

  pose_mod = PoseModel()
  pose_mod = geo.load_model(pose_mod, args.pose_model, device)

  gen_fnc = lambda z: flow_mod.g(torch.from_numpy(z.astype(np.float32)).to(device)).detach().cpu().numpy()
  enc_fnc = lambda x: flow_mod.f(torch.from_numpy(x.astype(np.float32)).to(device))[0].detach().cpu().numpy()
 
  pose_fnc = lambda x: pose_mod(x.to(device)).detach().cpu().numpy() 

  
  if args.display_flow:
    n_plots = 4
    # display generated poses
    hand_pose = gen_fnc(np.random.randn(n_plots, dim_in))
    
    fig = plt.figure()
    for i in range(n_plots):
      ax = fig.add_subplot('22{:d}'.format(i+1))
      geo.plot_skeleton2d(hand_pose[i], ax, autoscale=False)
      ax.set_xlim([-0.8, 0.8])
      ax.set_ylim([-0.8, 0.8])


  if args.display_pose:
    idx = np.random.randint(0, 1)
    print(idx)
    subject = np.random.choice(SUBJECTS)
    print(subject)
    gesture = np.random.choice(GESTURES)
    print(gesture)

    # regress poses and display
    ds = MSRADataset(subjects=[subject], gestures=[gesture], max_buffer=4, image=True)
    dl = DataLoader(ds, batch_size=500, shuffle=False)

    batch = dl.__iter__().__next__()
    pred_pose = pose_fnc(batch['img'])
    true_pose = batch['jts'].detach().cpu().numpy()
    img = batch['img'].detach().cpu().numpy()
    z_pose = enc_fnc(pred_pose)
    
    plt_img, plt_true, plt_pred = 131, 132, 133
    if args.filter_pose:
      dt = 0.1

      # Q = 15
      P, Q, R = 0.5, 20.0, 10.0
      z_param = {'P': P, 'Q': Q, 'R': R, 'dt': dt}
      x_param = {'P': P, 'Q': Q, 'R': R, 'dt': dt}

      z_smooth = kf.kalman_filter3d(z_pose, **z_param)
      latent_smoothed = gen_fnc(z_smooth)
      input_smoothed = kf.kalman_filter3d(pred_pose, **x_param)
      plt_img, plt_true, plt_pred, plt_zkf, plt_xkf = 231, 232, 234, 235, 236

    for i in range(400):
      fig = plt.figure()

      ax = fig.add_subplot(plt_pred)
      ax = geo.plot_skeleton2d(pred_pose[idx+i], ax, autoscale=False)
      geo.lim_axes(ax)
      ax.set_title('Predicted')

      ax = fig.add_subplot(plt_true)
      ax = geo.plot_skeleton2d(true_pose[idx+i], ax, autoscale=False)
      geo.lim_axes(ax)
      ax.set_title('True')

      ax = fig.add_subplot(plt_img)
      ax.imshow(np.clip(img[idx+i,0], 0.9, 1), cmap='Greys_r')
      ax.set_title('Depth image')
      
      if args.filter_pose:
        ax = fig.add_subplot(plt_zkf)
        ax = geo.plot_skeleton2d(latent_smoothed[idx+i], ax, autoscale=False)
        geo.lim_axes(ax)
        ax.set_title('Z KF smoothed')

        ax = fig.add_subplot(plt_xkf)
        ax = geo.plot_skeleton2d(input_smoothed[idx+i], ax, autoscale=False)
        geo.lim_axes(ax)
        ax.set_title('KF smoothed')

      plt.savefig('pose_{:03d}.png'.format(i))
      plt.close(fig)


  if args.neighborhood:
    n_neigh, idx = 8, 100
    std = 0.4
    subject = 'P1'
    seq = 'TIP'

    img, pose = get_hand(subject, seq, idx=idx)

    z = enc_fnc(pose[None,:])
    z_neigh = z + std * np.random.randn(n_neigh, z.shape[1])
    neighbors = gen_fnc(z_neigh) 
    
    fig = plt.figure()
    ax = fig.add_subplot(335)
    ax.imshow(np.clip(img, 0.9, 1), cmap='Greys_r')
    ax.set_title('Starting pose')

    for i, i_plt in enumerate([1,2,3,4,6,7,8,9]):
      ax = fig.add_subplot('33{}'.format(i_plt))
      ax = geo.plot_skeleton2d(neighbors[i], ax, autoscale=False)
      geo.lim_axes(ax)


  if args.interpolate:   
    n_interp, idx = 100, 130
    subject = 'P0'
    seq1 = '5'
    seq2 = '5'
    
    img1, pose1 = get_hand(subject, seq1, idx=idx)
    img2, pose2 = get_hand(subject, seq2, idx=idx)

    pose2 = geo.rotate(pose2.reshape((-1, 3)), np.pi, axis='y').flatten()

    # interp in latent space
    z = enc_fnc(geo.stack([pose1, pose2]))
    z_interp = geo.interpolate(z, n_interp) 
    latent_interp = gen_fnc(z_interp)

    # interp in data space
    x_interp = geo.interpolate(geo.stack([pose1, pose2]), n_interp)
 

    fig = plt.figure()

    ax = fig.add_subplot(121)
    ax.imshow(np.clip(img1, 0.9, 1), cmap='Greys_r')
    ax.set_title('Starting pose')

    ax = fig.add_subplot(122)
    ax.imshow(np.clip(img2, 0.9, 1), cmap='Greys_r')
    ax.set_title('Final pose')

    for i in range(n_interp):
      fig = plt.figure()
     
      ax = fig.add_subplot(221, projection='3d')
      ax = geo.plot_skeleton3d(latent_interp[i], ax, autoscale=False)
      geo.lim_axes3d(ax)
      ax.set_title('Latent interp')
      ax.view_init(elev=30, azim=30)

      ax = fig.add_subplot(222, projection='3d')
      ax = geo.plot_skeleton3d(x_interp[i], ax, autoscale=False)
      geo.lim_axes3d(ax)
      ax.set_title('Data interp')
      ax.view_init(elev=30, azim=30)

      ax = fig.add_subplot(223, projection='3d')
      ax = geo.plot_skeleton3d(latent_interp[i], ax, autoscale=False)
      geo.lim_axes3d(ax)
      ax.set_title('Latent interp')  
      ax.view_init(elev=80, azim=30)

      ax = fig.add_subplot(224, projection='3d')
      ax = geo.plot_skeleton3d(x_interp[i], ax, autoscale=False)
      geo.lim_axes3d(ax)
      ax.set_title('Data interp')
      ax.view_init(elev=80, azim=30)

      plt.savefig('interp_{:03d}.png'.format(i))
      plt.close(fig)


  if False:
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
