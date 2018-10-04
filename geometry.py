import os
import sys
import PIL
import colorsys
from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

import data


JOINTS = 'wrist index_mcp index_pip index_dip index_tip middle_mcp middle_pip middle_dip middle_tip ring_mcp ring_pip ring_dip ring_tip little_mcp little_pip little_dip little_tip thumb_mcp thumb_pip thumb_dip thumb_tip'.split(' ')

MS = 15
MS2 = 30

HandPart = namedtuple('HandPart', 'connections conn_color pts pt_color pt_size')

INDEX = HandPart(
  connections=np.array([[4,3], [3,2], [2,1]]),
  conn_color='g',
  pts=[1,2,3,4],
  pt_color='g',
  pt_size=MS)

MIDDLE = HandPart(
  connections=np.array([[8,7], [7,6], [6,5]]),
  conn_color='m',
  pts=[5,6,7,8],
  pt_color='m',
  pt_size=MS)

RING = HandPart(
  connections=np.array([[12,11], [11,10], [10,9]]),
  conn_color='r',
  pts=[9,10,11,12],
  pt_color='r',
  pt_size=MS)

LITTLE = HandPart(
  connections=np.array([[16,15], [15,14], [14,13]]),
  conn_color='c',
  pts=[13,14,15,16],
  pt_color='c',
  pt_size=MS)

THUMB = HandPart(
  connections=np.array([[20,19], [19,18], [18,17]]),
  conn_color='y',
  pts=[17,18,19,20],
  pt_color='y',
  pt_size=MS)

WRIST = HandPart(
  connections=np.array([[0,17], [0,1], [0,5], [0,9], [0,13]]),
  conn_color='b',
  pts=[0],
  pt_color='b',
  pt_size=MS2)

HAND = [INDEX, MIDDLE, RING, LITTLE, THUMB, WRIST]

lw_w = 3
lw_t1, lw_t2, lw_t3 = 2.8, 2.2, 2.1
lw_l1, lw_l2, lw_l3 = 2.3, 1.5, 1.2
lw_r1, lw_r2, lw_r3 = 2.4, 1.7, 1.4
lw_m1, lw_m2, lw_m3 = 2.7, 2.0, 1.8
lw_i1, lw_i2, lw_i3 = 2.8, 2.0, 1.8

s_w = 30
s_t1, s_t2, s_t3, s_t4 = 18, 16, 14, 12
s_l1, s_l2, s_l3, s_l4 = 18, 16, 14, 12
s_r1, s_r2, s_r3, s_r4 = 18, 16, 14, 12
s_m1, s_m2, s_m3, s_m4 = 18, 16, 14, 12
s_i1, s_i2, s_i3, s_i4 = 18, 16, 14, 12

OBJECTS = [
  ['l', 0, 17, lw_w],
  ['l', 0, 1, lw_w],
  ['l', 0, 5, lw_w],
  ['l', 0, 9, lw_w],
  ['l', 0, 13, lw_w],
  ['l', 20, 19, lw_t3],
  ['l', 19, 18, lw_t2],
  ['l', 18, 17, lw_t1],
  ['l', 16, 15, lw_l3],
  ['l', 15, 14, lw_l2],
  ['l', 14, 13, lw_l1],
  ['l', 12, 11, lw_r3],
  ['l', 11, 10, lw_r2],
  ['l', 10, 9, lw_r1],
  ['l', 8, 7, lw_m3],
  ['l', 7, 6, lw_m2],
  ['l', 6, 5, lw_m1],
  ['l', 4, 3, lw_i3],
  ['l', 3, 2, lw_i2],
  ['l', 2, 1, lw_i1],
  ['pt', 0, s_w, 2/3],  
  ['pt', 17, s_t1, 0],
  ['pt', 18, s_t2, 0],
  ['pt', 19, s_t3, 0],
  ['pt', 20, s_t4, 0],
  ['pt', 13, s_l1, 0.2],  
  ['pt', 14, s_l2, 0.2],
  ['pt', 15, s_l3, 0.2],
  ['pt', 16, s_l4, 0.2],
  ['pt', 9, s_r1, 0.4],
  ['pt', 10, s_r2, 0.4],
  ['pt', 11, s_r3, 0.4],
  ['pt', 12, s_r4, 0.4],
  ['pt', 5, s_m1, 0.6],
  ['pt', 6, s_m2, 0.6],
  ['pt', 7, s_m3, 0.6],
  ['pt', 8, s_m4, 0.6],
  ['pt', 1, s_i1, 0.8],
  ['pt', 2, s_i2, 0.8],
  ['pt', 3, s_i3, 0.8],
  ['pt', 4, s_i4, 0.8]]


def depth_to_camera(jts, azim, elev):
  L = 10
  camera = np.float32([[L*np.cos(azim), L*np.sin(azim), L*np.sin(elev)]])
  return np.sqrt(np.sum((jts - camera)**2, axis=1))


def depth_to_value(depth):
  depth_range = 0.4
  d = -depth
  d -= np.min(d)
  d = d/np.max(d)
  d *= depth_range
  d += (1-depth_range)
  return d


def plot_skeleton3d(jts, ax, azim=30, elev=60, autoscale=False, axes=False):
  if jts.shape[-1] != 3:
    jts = jts.reshape((-1, 3)) 

  depth = depth_to_camera(jts, azim*np.pi/180, elev*np.pi/180)
  
  depth_list = []
  for obj in OBJECTS:
    if obj[0] == 'l':
      depth_list += [np.mean(depth[obj[1:3]])]
    else:
      depth_list += [depth[obj[1]]]
  
  depth_arr = np.array(depth_list)
  furthest_idx = np.argsort(-depth_arr)
 
  #value_arr = depth_to_value(depth_arr)
  value_arr = 0.8*np.ones(depth_arr.shape)

  for i in furthest_idx:
    obj = OBJECTS[i]  
    depth = depth_arr[i]
    value = value_arr[i]
    
    if obj[0] == 'l':
      idx = obj[1:3]
      lw = obj[3]
      rgb = colorsys.hsv_to_rgb(2/3, 1, value)
      a = ax.plot(jts[idx,0], jts[idx,1], jts[idx,2])
      a[0].set_linewidth(lw) 
      a[0].set_color(rgb)

    else:
      idx = [obj[1]]
      rgb = colorsys.hsv_to_rgb(obj[3], 1, value)
      a = ax.plot(jts[idx,0], jts[idx,1], jts[idx,2], '.', markeredgecolor='none')
      a[0].set_markersize(obj[2])
      a[0].set_markerfacecolor(rgb)

  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.set_aspect('equal')

  return ax


def plot3d(x, ax, col='b', ms=10):
  ax.plot(x[:,0], x[:,1], x[:,2], '.'+col, markersize=ms)
  ax.set_xlabel('y')
  ax.set_ylabel('x')
  ax.set_zlabel('z')
  return ax


def old_plot_skeleton3d(x, ax, autoscale=True, axes=True):
  if x.shape[-1] != 3:
    x = x.reshape((-1, 3))

  for obj in HAND:
    ax = plot3d(x[obj.pts], ax, col=obj.pt_color, ms=obj.pt_size)
    for idx_pair in obj.connections:
      ax.plot(x[idx_pair, 0], x[idx_pair, 1], x[idx_pair, 2], obj.conn_color)

  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  ax.set_aspect('equal')

  if autoscale:
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
  ax.get_xaxis().set_visible(axes)
  ax.get_yaxis().set_visible(axes)
  return ax


def plot_skeleton2d(x, ax, autoscale=True, axes=False):
  if x.shape[-1] != 3:
    x = x.reshape((-1, 3))
  
  x[:,2] = -x[:,2]

  z = np.array([x[obj.pts[0], 2] for obj in HAND])
  plot_order = np.argsort(-z)
   
  for idx in plot_order:
    obj = HAND[idx]
    ax.plot(x[obj.pts, 0], x[obj.pts, 1], '.'+obj.pt_color, markersize=obj.pt_size)
    for idx_pair in obj.connections:
      ax.plot(x[idx_pair, 0], x[idx_pair, 1], obj.conn_color)

  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_aspect('equal')

  if autoscale:
    ranges = np.concatenate([ 
      np.min(x, axis=0)[None,:],
      np.max(x, axis=0)[None,:]], axis=0)

    max_range = np.ceil(np.max(ranges[1] - ranges[0]))
    mean_range = np.mean(ranges, axis=0)
   
    new_range = np.concatenate([
      (mean_range-max_range/2)[None,:],
      (mean_range+max_range/2)[None,:],
    ])
    
    ax.set_xlim(new_range[:,0])  
    ax.set_ylim(new_range[:,1])  
  ax.get_xaxis().set_visible(axes)
  ax.get_yaxis().set_visible(axes)
  return ax


def joints_over_depth(jts, img, ax):
  #ax.imshow(img.transpose(PIL.Image.FLIP_TOP_BOTTOM))
  ax.imshow(img[::-1,:], cmap='Greys_r')

  n_rows, n_cols = img.shape
  
  bounds = find_bounds(img)

  new_jts = jts.copy().reshape((-1,3))

  new_jts = normalize_dim(new_jts, bounds[3], bounds[1], 1)
  x_min = np.min(new_jts[:,0])
  new_jts[:,0] += bounds[0] - x_min
  
  plot_skeleton2d(new_jts, ax, autoscale=False)
  return ax


def find_bounds(img, thresh=0.7):
  truth = img > thresh  
  col_bounds = np.arange(img.shape[1])[np.any(truth, axis=0)][[0,-1]]
  row_bounds = np.arange(img.shape[0])[np.any(truth, axis=1)][[0,-1]]
  return col_bounds[0], img.shape[0]-row_bounds[0], col_bounds[1], img.shape[0]-row_bounds[1]


def normalize_dim(x, goal_min, goal_max, dim):
  # put in [0,1]
  min_val = np.min(x[:,dim])
  x -= min_val
  max_val = np.max(x[:,dim])
  x = x/max_val

  mag = goal_max - goal_min
  x *= mag
  x[:,dim] += goal_min
  
  return x
  
 
def get_quaternion(jts):
  v1 = jts[1] - jts[0]
  v2 = jts[13] - jts[0]
  
  out = np.float32([
    v1[1]*v2[2]-v2[1]*v1[2], 
    v1[0]*v2[2]-v2[0]*v1[2], 
    v1[0]*v2[1]-v2[0]*v1[1]])

  heading = np.arctan2(out[[1]], out[[0]])[0]
  attitude = np.arctan2(out[[2]], out[[0]])[0] - np.pi/2

  c1 = np.cos(heading/2)
  c2 = np.cos(attitude/2)
  c3 = 1
  s1 = np.sin(heading/2)
  s2 = np.sin(attitude/2)
  s3 = 0
  
  q = np.float32([
    c1*c2*c3 - s1*s2*s3,
    s1*s2*c3 + c1*c2*s3,
    s1*c2*c3 + c1*s2*s3,
    c1*s2*c3 - s1*c2*s3])
   
  return q 


def get_angles(jts):
  v1 = jts[1] - jts[0]
  v2 = jts[13] - jts[0]
  
  out = np.float32([
    v1[1]*v2[2]-v2[1]*v1[2], 
    v1[0]*v2[2]-v2[0]*v1[2], 
    v1[0]*v2[1]-v2[0]*v1[1]])

  azim = np.arctan2(out[[1]], out[[0]])
  elev = np.arctan2(out[[2]], out[[0]])
  return azim, elev  


def fix_2pi(z):
  new_z = z.copy()
  new_z[1,0] = closer_angle(z[0,0], z[1,0])
  new_z[1,1] = closer_angle(z[0,1], z[1,1])
  return new_z


def closer_angle(ang1, ang2, k_max=30):
  ang_candidates = 2*np.pi*np.arange(-k_max, k_max+1) + ang2
  return ang_candidates[np.argmin(np.abs(ang1 - ang_candidates))]


def hamilton_product(q1, q2):
  q = np.concatenate([
    q1[:,[0]]*q2[:,[0]] - q1[:,[1]]*q2[:,[1]] - q1[:,[2]]*q2[:,[2]] - q1[:,[3]]*q2[:,[3]], 
    q1[:,[0]]*q2[:,[1]] + q1[:,[1]]*q2[:,[0]] + q1[:,[2]]*q2[:,[3]] - q1[:,[3]]*q2[:,[2]], 
    q1[:,[0]]*q2[:,[2]] - q1[:,[1]]*q2[:,[3]] + q1[:,[2]]*q2[:,[0]] + q1[:,[3]]*q2[:,[1]], 
    q1[:,[0]]*q2[:,[3]] + q1[:,[1]]*q2[:,[2]] - q1[:,[2]]*q2[:,[1]] + q1[:,[3]]*q2[:,[0]]], axis=1)
  return q


def quaternion_rotation(q, v):
  q = q.reshape((1, -1))
  u = np.concatenate([np.zeros((v.shape[0], 1)), v], axis=1).astype(np.float32)
  mask = np.float32([[1, -1, -1, -1]])
  u_new = hamilton_product(u, mask*q)
  return hamilton_product(q, u_new)[:, 1:]


def random_quaternion():
  """Taken from 'Planning Algorithms', Stven LaValle, Eq. 5.15"""
  u1, u2, u3 = np.random.rand(3)
  u1sq1 = np.sqrt(1-u1)
  u1sq2 = np.sqrt(u1)

  q = np.float32([
    u1sq1*np.sin(2*np.pi*u2), 
    u1sq1*np.cos(2*np.pi*u2),
    u1sq2*np.sin(2*np.pi*u3),
    u1sq2*np.cos(2*np.pi*u3)]).astype(np.float32)
  
  return q


def rand_rotate(x, theta, az, el):
  """axis angle rotation"""
  x = np.cos(az) * np.cos(el)
  y = np.sin(az) * np.cos(el)
  z = np.sin(el)

  R = np.float32([
    [np.cos(theta)+x*x*(1-np.cos(theta)), x*y*(1-np.cos(theta))-z*np.sin(theta), x*z*(1-np.cos(theta))+y*np.sin(theta)],
    [y*x*(1-np.cos(theta))+z*np.sin(theta), np.cos(theta)+y*y*(1-np.cos(theta)), y*z*(1-np.cos(theta))-x*np.sin(theta)],
    [z*x*(1-np.cos(theta))-y*np.sin(theta), z*y*(1-np.cos(theta))+x*np.sin(theta), np.cos(theta)+z*z*(1-np.cos(theta))]])

  return np.dot(R, x.transpose()).transpose().astype(np.float32)


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


def load_model(mod, mod_file, device):
  if device.type == 'cpu':
    mod.load_state_dict(torch.load(mod_file, map_location='cpu'))
  else:
    mod.load_state_dict(torch.load(mod_file))

  mod.to(device)
  mod.eval()
  return mod


def lim_axes(ax, lim=[-1, 1]):
  ax.set_xlim(lim)
  ax.set_ylim(lim)


def lim_axes3d(ax, lim=[-0.8, 0.8]):
  ax.set_xlim(lim)
  ax.set_ylim(lim)
  ax.set_zlim(lim)
 

def stack(lst):
  lst = [jts.reshape((1,-1)) for jts in lst]
  return np.concatenate(lst, axis=0)


def interpolate(x, n):
  out = np.zeros((n, x.shape[1])).astype(np.float32)
  wts = np.linspace(0, 1, n)
  for i, wt in enumerate(wts):
    out[i] = x[0] + wt*(x[1] - x[0])
  return out


if __name__ == '__main__':
  
  img, jts = data.get_hand('P1', '5')
  
  q = get_quaternion(jts.reshape((-1, 3)))
  
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  plot_skeleton3d(jts, ax)

  plt.show()
