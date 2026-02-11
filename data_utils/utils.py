import numpy as np


def get_class_weights(dataset, log=False, **kwargs):
    targets = dataset.y

    class_sample_count = np.unique(targets, return_counts=True)[1]
    weights = 1. / class_sample_count

    min_w = weights.min()
    weights = weights / min_w

    if log:
        weights = np.log(weights)

    return np.clip(weights, a_min=1., a_max=None)


def construct_balanced_subset(x, y):
    xdata, ydata = [], []
    minsize = np.inf
    for cls_ in np.unique(y):
        xdata.append(x[y == cls_])
        ydata.append(y[y == cls_])
        if ydata[-1].shape[0] < minsize:
            minsize = ydata[-1].shape[0]
    for i in range(len(xdata)):
        if xdata[i].shape[0] < minsize:
            import pdb
            pdb.set_trace()
        idx = np.arange(xdata[i].shape[0])
        np.random.shuffle(idx)
        xdata[i] = xdata[i][idx][:minsize]
        ydata[i] = ydata[i][idx][:minsize]
    # !list
    return np.concatenate(xdata, 0), np.concatenate(ydata, 0)


def downsample_subset(x, y, sample_size=500):
    xdata, ydata = [], []
    for cls_ in np.unique(y):
        xdata.append(x[y == cls_])
        ydata.append(y[y == cls_])
        sample_size = min(sample_size, ydata[-1].shape[0])
    for i in range(len(xdata)):
        idx = np.arange(xdata[i].shape[0])
        np.random.shuffle(idx)
        xdata[i] = xdata[i][idx][:sample_size]
        ydata[i] = ydata[i][idx][:sample_size]

    return np.concatenate(xdata, 0), np.concatenate(ydata, 0)
