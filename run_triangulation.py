import numpy as np
from dataset import load_sfm_dataset
from normalize import normalize_image_pts
from triangulation import solve_triangulation

dataset = load_sfm_dataset()
inv_k = np.linalg.inv(dataset.k)
xs = [normalize_image_pts(x, inv_k) for x in dataset.xs]

print(dataset.r)
print(dataset.c)

r = dataset.r.transpose()
t = -r @ dataset.c
projection = np.concatenate((r, t), axis=-1)

coordinates = [solve_triangulation(x, y, projection)
               for x, y in zip(xs[0], xs[1])]

print(xs[0][0])

print(coordinates[0])
