import numbers
import array
import numpy as np
from scipy import linalg
import scipy.sparse as sp
from collections import Iterable
from ..preprocessing import MultiLabelBinarizer
from ..utils import check_array, check_random_state
from ..utils import shuffle as util_shuffle
from ..utils.random import sample_without_replacement
from ..externals import six
map = six.moves.map
zip = six.moves.zip

def make_blobs(n_samples=100, n_features=2, centers=None, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None):
    generator = check_random_state(random_state)
    if isinstance(n_samples, numbers.Integral):
        if centers is None:
            centers = 3
        if isinstance(centers, numbers.Integral):
            n_centers = centers
            centers = generator.uniform(center_box[0], center_box[1], size=(n_centers, n_features))
        else:
            centers = check_array(centers)
            n_features = centers.shape[1]
            n_centers = centers.shape[0]
    else:
        n_centers = len(n_samples)
        if centers is None:
            centers = generator.uniform(center_box[0], center_box[1], size=(n_centers, n_features))
        try:
            assert len(centers) == n_centers
        except TypeError:
            raise ValueError('Parameter `centers` must be array-like. Got {!r} instead'.format(centers))
        except AssertionError:
            raise ValueError('Length of `n_samples` not consistent with number of centers. Got n_samples = {} and centers = {}'.format(n_samples, centers))
        else:
            centers = check_array(centers)
            n_features = centers.shape[1]
    if hasattr(cluster_std, '__len__') and len(cluster_std) != n_centers:
        raise ValueError('Length of `clusters_std` not consistent with number of centers. Got centers = {} and cluster_std = {}'.format(centers, cluster_std))
    if isinstance(cluster_std, numbers.Real):
        cluster_std = np.ones(len(centers)) * cluster_std
    X = []
    y = []
    if isinstance(n_samples, Iterable):
        n_samples_per_center = n_samples
    else:
        n_samples_per_center = [int(n_samples // n_centers)] * n_centers
        for i in range(n_samples % n_centers):
            n_samples_per_center[i] += 1
    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        X.append(generator.normal(loc=centers[i], scale=std, size=(n, n_features)))
        y += [i] * n
    X = np.concatenate(X)
    y = np.array(y)
    if shuffle:
        total_n_samples = np.sum(n_samples)
        indices = np.arange(total_n_samples)
        generator.shuffle(indices)
        X = X[indices]
        y = y[indices]
    return (X, y)