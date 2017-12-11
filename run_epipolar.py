import epipolar
import numpy as np
from dataset import load_sfm_dataset
from normalize import normalize_image_pts

dataset = load_sfm_dataset()
inv_k = np.linalg.inv(dataset.k)
xs = [normalize_image_pts(x, inv_k) for x in dataset.xs]

pairs = np.array((xs[0], xs[1])).transpose((1, 0, 2))

essential = epipolar.estimate_essential(pairs)



# second image's extrinsic parameters
r, c = epipolar.estimate_euclidean(essential, pairs)

left, right = pairs[100]

print(left @ essential @ right)
