import os
import numpy as np
from matplotlib import pyplot as plt
from dataset import visualize_dataset
from dataset import load_dataset
from dlt_homography import solve_dlt


path = '/Users/pu/projects/logo_projection/RoboticsPerceptionWeek2AssignmentCode'

logo_path = os.path.join(path, 'images/logos/penn_engineering_logo.png')

logo = plt.imread(logo_path)

dataset = load_dataset(path)

y_size, x_size = logo.shape[:2]

src_pts = [(0, 0), (0, y_size-1), (x_size, 0), (x_size, y_size)]

transforms = []


pts = dataset.pts.transpose(2, 0, 1)
# transform plane to target location

print(pts[0])
transforms.append(solve_dlt(zip(src_pts, pts[0])))

# combine the two images
mul = np.matmul(transforms[0], [0,0,1])
print(mul / mul[-1], pts[0][0])
