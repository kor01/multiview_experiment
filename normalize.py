import numpy as np

def normalize_image_pts(pts, inv_k):
  batch_size = pts.shape[0]
  ones = np.ones((batch_size, 1), dtype='float64')
  pts = np.concatenate((pts, ones), axis=-1)
  pts = pts.transpose()
  pts = inv_k @ pts
  pts = np.transpose(pts)
  return pts
