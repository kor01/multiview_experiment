import os
from matplotlib import pyplot as plt
from dataset import visualize_dataset
from dataset import load_dataset
from dlt_homography import solve_dlt


path = '/Users/pu/projects/logo_projection/RoboticsPerceptionWeek2AssignmentCode'

logo_path = os.path.join(path, 'images/logos/penn_engineering_logo.png')

logo = plt.imread(logo_path)

dataset = load_dataset(path)

y_size, x_size = logo.shape

src_pts = [(0, 0), (0, y_size-1), (x_size, 0), (x_size, y_size)]

transforms = []

# transform plane to target location
for pts in dataset.pts:
  transforms.append(solve_dlt(zip(src_pts, pts)))

# combine the two images
