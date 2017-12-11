import numpy as np


def ransac(model, metric, minimal,
           err_threshold, ratio_threshold, dataset):
  size = len(dataset)

  while True:
    model.reset()
    sample = np.random.choice(
      len(dataset), minimal, replace=False)
    model.fit(sample)
    values = model.predict(dataset[:][0])
    err = metric(values, dataset[:][1])
    inliners = err < err_threshold
    if float(sum(inliners)) / size > ratio_threshold:
      model.fit(dataset[inliners])
      return

