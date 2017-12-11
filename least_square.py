"""
solve least square problem of various kind
"""

import numpy as np

def solve_linear_homogeneous(coeff):
  u, d, v = np.linalg.svd(coeff)
  return v[-1, :]


