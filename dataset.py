import os
import glob
import numpy as np
from collections import namedtuple
from scipy.io import loadmat
from matplotlib import pyplot as plt


Dataset = namedtuple('Dataset', ('images', 'pts'))


def load_dataset(path):
  image_path = glob.glob(os.path.join(path, 'images/barcaReal/*.jpg'))
  pts = loadmat(os.path.join(path, 'data/BarcaReal_pts.mat'))['video_pts']
  images = tuple([plt.imread(x) for x in image_path])
  return Dataset(images=images, pts=pts)


def visualize_dataset(dataset):
  pts = dataset.pts.transpose((2, 0, 1))
  ret = []
  for image, pt in zip(dataset.images, pts):
    coords_x = pt[:, 0].astype('int32')
    coords_y = pt[:, 1].astype('int32')

    image = image.copy()
    for i in range(2):
      for j in range(2):
        y = np.clip(coords_y + i, 0, image.shape[0] - 1)
        x = np.clip(coords_x + j, 0, image.shape[1] - 1)
        image[y, x, :] = 0
    ret.append(image)

  return ret


SFM_PATH = '/Users/pu/work/PycharmProjects/multiview_experiment/data/data.mat'
#SFM_PATH = '/home/pu/PycharmProjects/multiview_experiment/data/data.mat'

SFMDataset = namedtuple(
  'SFMDataset', ('xs', 'imgs', 'k', 'r', 'c'))


def load_sfm_dataset(path=SFM_PATH):
  dataset = loadmat(path)
  dataset = dataset['data']
  images = [dataset['img%d' % i][0][0] for i in range(1, 4)]
  xs = [dataset['x%d' % i][0][0] for i in range(1, 4)]
  k, r, c = dataset['K'][0][0], dataset['R'][0][0], dataset['C'][0][0]
  return SFMDataset(xs=xs, imgs=images, k=k, r=r, c=c)
