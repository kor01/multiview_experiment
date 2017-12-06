from matplotlib import pyplot as plt
from dataset import visualize_dataset
from dataset import load_dataset

path = '/Users/pu/projects/logo_projection/RoboticsPerceptionWeek2AssignmentCode'

dataset = load_dataset(path)

visual = visualize_dataset(dataset)

plt.imshow(visual[0])
plt.show()
