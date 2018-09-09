import os
import sys
from glob import glob
import numpy as np


def stack(lst):
  lst = [jts.reshape((1,-1)) for jts in lst]
  return np.concatenate(lst, axis=0)

def interpolate(x, n):
  out = np.zeros((n, x.shape[1])).astype(np.float32)
  wts = np.linspace(0, 1, n)
  for i, wt in enumerate(wts):
    out[i] = x[0] + wt*(x[1] - x[0])
  return out
