import numpy as np
from pnp import solve_pnp
from dataset import load_sfm_dataset
from normalize import normalize_image_pts
from triangulation import solve_triangulation


dataset = load_sfm_dataset()
inv_k = np.linalg.inv(dataset.k)
xs = [normalize_image_pts(x, inv_k) for x in dataset.xs]


r = dataset.r.transpose()
t = -r @ dataset.c
projection = np.concatenate((r, t), axis=-1)

coordinates = [solve_triangulation(x, y, projection)
               for x, y in zip(xs[0], xs[1])]

idx = 100
rec = r @ coordinates[idx] + t.ravel()
print(coordinates[idx].shape)
print('validate:', xs[1][idx], rec / rec[2])

r, t = solve_pnp(list(zip(xs[2], coordinates)))

print('shape:', r, t)

#r, t = extrinsic[:, :3], extrinsic[:, -1]

rec = r @ coordinates[idx] + t.ravel()

print(rec / rec[2], xs[2][idx])
