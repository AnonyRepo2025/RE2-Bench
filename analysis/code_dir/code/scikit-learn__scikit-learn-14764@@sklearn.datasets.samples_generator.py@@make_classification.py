import numbers
import array
from collections.abc import Iterable
import numpy as np
from scipy import linalg
import scipy.sparse as sp
from ..preprocessing import MultiLabelBinarizer
from ..utils import check_array, check_random_state
from ..utils import shuffle as util_shuffle
from ..utils.random import sample_without_replacement

def make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None):
    generator = check_random_state(random_state)
    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError('Number of informative, redundant and repeated features must sum to less than the number of total features')
    if n_informative < np.log2(n_classes * n_clusters_per_class):
        msg = 'n_classes({}) * n_clusters_per_class({}) must be'
        msg += ' smaller or equal 2**n_informative({})={}'
        raise ValueError(msg.format(n_classes, n_clusters_per_class, n_informative, 2 ** n_informative))
    if weights is not None:
        if len(weights) not in [n_classes, n_classes - 1]:
            raise ValueError('Weights specified but incompatible with number of classes.')
        if len(weights) == n_classes - 1:
            if isinstance(weights, list):
                weights = weights + [1.0 - sum(weights)]
            else:
                weights = np.resize(weights, n_classes)
                weights[-1] = 1.0 - sum(weights[:-1])
    else:
        weights = [1.0 / n_classes] * n_classes
    n_useless = n_features - n_informative - n_redundant - n_repeated
    n_clusters = n_classes * n_clusters_per_class
    n_samples_per_cluster = [int(n_samples * weights[k % n_classes] / n_clusters_per_class) for k in range(n_clusters)]
    for i in range(n_samples - sum(n_samples_per_cluster)):
        n_samples_per_cluster[i % n_clusters] += 1
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=np.int)
    centroids = _generate_hypercube(n_clusters, n_informative, generator).astype(float, copy=False)
    centroids *= 2 * class_sep
    centroids -= class_sep
    if not hypercube:
        centroids *= generator.rand(n_clusters, 1)
        centroids *= generator.rand(1, n_informative)
    X[:, :n_informative] = generator.randn(n_samples, n_informative)
    stop = 0
    for k, centroid in enumerate(centroids):
        start, stop = (stop, stop + n_samples_per_cluster[k])
        y[start:stop] = k % n_classes
        X_k = X[start:stop, :n_informative]
        A = 2 * generator.rand(n_informative, n_informative) - 1
        X_k[...] = np.dot(X_k, A)
        X_k += centroid
    if n_redundant > 0:
        B = 2 * generator.rand(n_informative, n_redundant) - 1
        X[:, n_informative:n_informative + n_redundant] = np.dot(X[:, :n_informative], B)
    if n_repeated > 0:
        n = n_informative + n_redundant
        indices = ((n - 1) * generator.rand(n_repeated) + 0.5).astype(np.intp)
        X[:, n:n + n_repeated] = X[:, indices]
    if n_useless > 0:
        X[:, -n_useless:] = generator.randn(n_samples, n_useless)
    if flip_y >= 0.0:
        flip_mask = generator.rand(n_samples) < flip_y
        y[flip_mask] = generator.randint(n_classes, size=flip_mask.sum())
    if shift is None:
        shift = (2 * generator.rand(n_features) - 1) * class_sep
    X += shift
    if scale is None:
        scale = 1 + 100 * generator.rand(n_features)
    X *= scale
    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)
        indices = np.arange(n_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]
    return (X, y)