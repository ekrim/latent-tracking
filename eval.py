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

import kf
import flows
import geometry as geo
from data import MSRADataset, SUBJECTS, GESTURES, get_hand
from model import RealNVP, PoseModel


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--flow_model', default='models/flow_model.pytorch', help='trained RealNVP model')
  parser.add_argument('--pose_model', default='models/pose_model.pytorch', help='trained pose model')
  parser.add_argument('--dim_in', default=63, type=int, help='dimensionality of input data')
  parser.add_argument('--model_type', default='maf', help='maf or realnvp')
  parser.add_argument('--display_pose', action='store_true', help='display pose estimates')
  parser.add_argument('--filter_pose', action='store_true', help='kalman filter pose smoothing')
  parser.add_argument('--display_flow', action='store_true', help='generate synthetic poses and display them')
  parser.add_argument('--interpolate', action='store_true', help='run interpolation experiment')
  parser.add_argument('--interpolate_full', action='store_true', help='run interpolation through multiple keyframes')
  parser.add_argument('--neighborhood', action='store_true', help='run neighborhood generation')
  parser.add_argument('--inverse_kinematics', action='store_true', help='run inverse kinematics experiment')
  args = parser.parse_args(sys.argv[1:])

  dim_in = args.dim_in
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  if args.model_type == 'realnvp':
    flow_mod = RealNVP(args.dim_in, device, n_hid=256, n_mask=10)

  elif args.model_type == 'maf':
    flow_mod = flows.FlowSequential(10, args.dim_in, 256, device)

  else:
    raise ValueError('no such flow')

  flow_mod = geo.load_model(flow_mod, args.flow_model, device)

  pose_mod = PoseModel()
  pose_mod = geo.load_model(pose_mod, args.pose_model, device)
  pose_mod.eval()

  pose_mod2 = PoseModel()
  pose_mod2 = geo.load_model(pose_mod2, 'models/pose_model_z.pytorch', device)
  pose_mod2.eval()

  gen_fnc = lambda z: flow_mod.g(torch.from_numpy(z.astype(np.float32)).to(device)).detach().cpu().numpy()
  enc_fnc = lambda x: flow_mod.forward(torch.from_numpy(x.astype(np.float32)).to(device))[0].detach().cpu().numpy()
  logp_fnc = lambda x: flow_mod.log_prob(torch.from_numpy(x.astype(np.float32)).to(device)).detach().cpu().numpy()
 
  pose_fnc = lambda x: pose_mod(x.to(device)).detach().cpu().numpy() 
  pose_fnc2 = lambda x: pose_mod2(x.to(device)).detach().cpu().numpy() 

  
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
    #subject = np.random.choice(SUBJECTS)
    subject = 'P8'
    print(subject)
    gesture = np.random.choice(GESTURES)
    print(gesture)

    # regress poses and display
    ds = MSRADataset(subjects=[subject], gestures=[gesture], max_buffer=4, image=True)
    dl = DataLoader(ds, batch_size=5, shuffle=False)

    batch = dl.__iter__().__next__()
    pred_pose = pose_fnc(batch['img'])
    pred_pose_z = pose_fnc2(batch['img'])
     
    true_pose = batch['jts'].detach().cpu().numpy()
    L = true_pose.shape[0] 
    img = batch['img'].detach().cpu().numpy()
    z_pose = enc_fnc(pred_pose)
    
    plt_img, plt_true, plt_pred = 131, 132, 133
    if args.filter_pose:
      dt = 0.1

      # Q = 15
      P, Q, R = 0.5, 100.0, 10.0
      z_param = {'P': P, 'Q': Q, 'R': R, 'dt': dt}
      x_param = {'P': P, 'Q': Q, 'R': R, 'dt': dt}

      z_smooth = kf.kalman_filter3d(z_pose, **z_param)
      latent_smoothed = gen_fnc(z_smooth)
      input_smoothed = kf.kalman_filter3d(pred_pose, **x_param)
 
      pred_z_smooth = kf.kalman_filter3d(pred_pose_z, **z_param)
      pred_z_smooth = gen_fnc(pred_z_smooth)
      pred_z_orig = gen_fnc(pred_pose_z)

      plt_img, plt_true, plt_pred, plt_xkf, plt_predz, plt_zkf = 322, 321, 323, 324, 325, 326

    z_mse = np.mean(((true_pose - pred_z_smooth)**2).flatten())
    x_mse = np.mean(((true_pose - input_smoothed)**2).flatten())
    print('z smoothed error: {}'.format(z_mse))
    print('x smoothed error: {}'.format(x_mse))
    for i in range(L):
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
        ax = fig.add_subplot(plt_predz)
        ax = geo.plot_skeleton2d(pred_z_orig[idx+i], ax, autoscale=False)
        geo.lim_axes(ax)
        ax.set_title('Pred Z')

        ax = fig.add_subplot(plt_zkf)
        ax = geo.plot_skeleton2d(pred_z_smooth[idx+i], ax, autoscale=False)
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
    elev1, elev2 = 0, 90
    azim1, azim2 = 30, 30

    n_interp, idx = 100, 130
    subject = 'P0'
    seq1 = '5'
    seq2 = '5'
    
    img1, pose1 = get_hand(subject, seq1, idx=idx)
    img2, pose2 = get_hand(subject, seq2, idx=idx)

    pose2 = geo.rotate(pose2.reshape((-1, 3)), np.pi, axis='y').flatten()

    # interp in latent space
    z = enc_fnc(geo.stack([pose1, pose2]))
   
    print(z[:,:2])
    #z = geo.fix_2pi(z)
    print(z[:,:2])
   
    z_interp = geo.interpolate(z, n_interp) 
    latent_interp = gen_fnc(z_interp)

    # plot prob trajectory
    #probs = logp_fnc(latent_interp).flatten()

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.plot(probs)
    #ax.set_xlabel('Interp step')
    #ax.set_ylabel('log p(x)')
    #ax.set_title('Sample likelihood over interpolation')

    zeros_hand = gen_fnc(np.zeros((1,63)))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax = geo.plot_skeleton3d(zeros_hand, ax, autoscale=False)
    geo.lim_axes3d(ax)
    ax.set_title('0 hand')
    

    # interp in data space
    x_interp = geo.interpolate(geo.stack([pose1, pose2]), n_interp)

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.imshow(np.clip(img1, 0.9, 1), cmap='Greys_r')
    ax.set_title('Hand to interpolate')

    for i in range(n_interp):
      fig = plt.figure()
     
      ax = fig.add_subplot(221, projection='3d')
      ax = geo.plot_skeleton3d(latent_interp[i], ax, autoscale=False)
      geo.lim_axes3d(ax)
      ax.set_title('Latent interp')
      ax.view_init(elev=elev1, azim=azim1)

      ax = fig.add_subplot(222, projection='3d')
      ax = geo.plot_skeleton3d(x_interp[i], ax, autoscale=False)
      geo.lim_axes3d(ax)
      ax.set_title('Data interp')
      ax.view_init(elev=elev1, azim=azim1)

      ax = fig.add_subplot(223, projection='3d')
      ax = geo.plot_skeleton3d(latent_interp[i], ax, autoscale=False)
      geo.lim_axes3d(ax)
      ax.set_title('Latent interp')  
      ax.view_init(elev=elev2, azim=azim2)

      ax = fig.add_subplot(224, projection='3d')
      ax = geo.plot_skeleton3d(x_interp[i], ax, autoscale=False)
      geo.lim_axes3d(ax)
      ax.set_title('Data interp')
      ax.view_init(elev=elev2, azim=azim2)

      plt.savefig('interp_{:03d}.png'.format(i))
      plt.close(fig)


  if args.interpolate_full:   
    elev1, elev2 = 0, 90
    azim1, azim2 = 30, 30

    n_interp, idx = 50, 130
    subject = 'P0'
    gestures = [
      ('5',     0,     0,     0),
      ('IP',    0, 0,     0),
      ('3', 0,     0,     0),
      ('MP',    0,     0, 0)]

    img_list = []
    for i, gest in enumerate(gestures):
      axis_angles = [(('x', 'y', 'z')[i_xyz], ang) for i_xyz, ang in enumerate(gest[1:]) if ang != 0]

      img2, pose2 = get_hand(subject, gest[0], idx=idx)
      img_list += [img2]
      
      for ax_ang in axis_angles:
        pose2 = geo.rotate(pose2.reshape((-1, 3)), ax_ang[1], axis=ax_ang[0]).flatten() 

      if i > 0:
        z = enc_fnc(geo.stack([pose1, pose2]))
        z_interp += [geo.interpolate(z, n_interp)[1:]]
      else:
        z_interp = [pose2.reshape((1, -1))]
      
      pose1 = pose2

    latent_interp = gen_fnc(np.concatenate(z_interp, axis=0))


    fig = plt.figure()
    for i, img in enumerate(img_list):
      ax = fig.add_subplot('14{:d}'.format(i+1))
      ax.imshow(np.clip(img, 0.9, 1), cmap='Greys_r')
      ax.set_title('key frame {:d} (unrotated)'.format(i+1))

    for i in range(latent_interp.shape[0]):
      fig = plt.figure()
     
      ax = fig.add_subplot(211, projection='3d')
      ax = geo.plot_skeleton3d(latent_interp[i], ax, autoscale=False)
      geo.lim_axes3d(ax)
      ax.set_title('Latent interp')
      ax.view_init(elev=elev1, azim=azim1)

      ax = fig.add_subplot(212, projection='3d')
      ax = geo.plot_skeleton3d(latent_interp[i], ax, autoscale=False)
      geo.lim_axes3d(ax)
      ax.set_title('Latent interp')  
      ax.view_init(elev=elev2, azim=azim2)

      plt.savefig('interp_{:03d}.png'.format(i))
      plt.close(fig)


  if args.inverse_kinematics:
    # run inverse kinematics experiments
    idx = 100
    subject = 'P1'
    seq = 'TIP'

    img, pose = get_hand(subject, seq, idx=idx)
    
    jts = pose.reshape((-1, 3))
    bad_jts = jts.copy()
    bad_jts[8] = jts[20].copy()
    bad_jts[20] = jts[8].copy()
 
    z_bad = enc_fnc(bad_jts.reshape((1, -1)))
    z_bad *= 0.7
    x_bad = gen_fnc(z_bad)
   
    fig = plt.figure()
    ax = fig.add_subplot(131, projection='3d')
    ax = geo.plot_skeleton3d(bad_jts, ax)
    ax = fig.add_subplot(132, projection='3d')
    ax = geo.plot_skeleton3d(x_bad, ax)
    ax = fig.add_subplot(133, projection='3d')
    ax = geo.plot_skeleton3d(jts, ax)
    

  plt.show() 
