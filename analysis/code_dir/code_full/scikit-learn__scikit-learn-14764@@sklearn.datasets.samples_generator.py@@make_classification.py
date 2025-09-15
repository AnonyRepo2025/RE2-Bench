def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState instance' % seed)

def _generate_hypercube(samples, dimensions, rng):
    if dimensions > 30:
        return np.hstack([rng.randint(2, size=(samples, dimensions - 30)), _generate_hypercube(samples, 30, rng)])
    out = sample_without_replacement(2 ** dimensions, samples, random_state=rng).astype(dtype='>u4', copy=False)
    out = np.unpackbits(out.view('>u1')).reshape((-1, 32))[:, -dimensions:]
    return out

def shuffle(*arrays, **options):
    options['replace'] = False
    return resample(*arrays, **options)

def resample(*arrays, **options):
    random_state = check_random_state(options.pop('random_state', None))
    replace = options.pop('replace', True)
    max_n_samples = options.pop('n_samples', None)
    stratify = options.pop('stratify', None)
    if options:
        raise ValueError('Unexpected kw arguments: %r' % options.keys())
    if len(arrays) == 0:
        return None
    first = arrays[0]
    n_samples = first.shape[0] if hasattr(first, 'shape') else len(first)
    if max_n_samples is None:
        max_n_samples = n_samples
    elif max_n_samples > n_samples and (not replace):
        raise ValueError('Cannot sample %d out of arrays with dim %d when replace is False' % (max_n_samples, n_samples))
    check_consistent_length(*arrays)
    if stratify is None:
        if replace:
            indices = random_state.randint(0, n_samples, size=(max_n_samples,))
        else:
            indices = np.arange(n_samples)
            random_state.shuffle(indices)
            indices = indices[:max_n_samples]
    else:
        y = check_array(stratify, ensure_2d=False, dtype=None)
        if y.ndim == 2:
            y = np.array([' '.join(row.astype('str')) for row in y])
        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]
        class_counts = np.bincount(y_indices)
        class_indices = np.split(np.argsort(y_indices, kind='mergesort'), np.cumsum(class_counts)[:-1])
        n_i = _approximate_mode(class_counts, max_n_samples, random_state)
        indices = []
        for i in range(n_classes):
            indices_i = random_state.choice(class_indices[i], n_i[i], replace=replace)
            indices.extend(indices_i)
        indices = random_state.permutation(indices)
    arrays = [a.tocsr() if issparse(a) else a for a in arrays]
    resampled_arrays = [safe_indexing(a, indices) for a in arrays]
    if len(resampled_arrays) == 1:
        return resampled_arrays[0]
    else:
        return resampled_arrays

def check_consistent_length(*arrays):
    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError('Found input variables with inconsistent numbers of samples: %r' % [int(l) for l in lengths])

def _num_samples(x):
    message = 'Expected sequence or array-like, got %s' % type(x)
    if hasattr(x, 'fit') and callable(x.fit):
        raise TypeError(message)
    if not hasattr(x, '__len__') and (not hasattr(x, 'shape')):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError(message)
    if hasattr(x, 'shape') and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError('Singleton array %r cannot be considered a valid collection.' % x)
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]
    try:
        return len(x)
    except TypeError:
        raise TypeError(message)

def safe_indexing(X, indices, axis=0):
    if axis == 0:
        return _safe_indexing_row(X, indices)
    elif axis == 1:
        return _safe_indexing_column(X, indices)
    else:
        raise ValueError("'axis' should be either 0 (to index rows) or 1 (to index  column). Got {} instead.".format(axis))

def _safe_indexing_row(X, indices):
    if hasattr(X, 'iloc'):
        indices = np.asarray(indices)
        indices = indices if indices.flags.writeable else indices.copy()
        try:
            return X.iloc[indices]
        except ValueError:
            warnings.warn('Copying input dataframe for slicing.', DataConversionWarning)
            return X.copy().iloc[indices]
    elif hasattr(X, 'shape'):
        if hasattr(X, 'take') and (hasattr(indices, 'dtype') and indices.dtype.kind == 'i'):
            return X.take(indices, axis=0)
        else:
            return _array_indexing(X, indices, axis=0)
    else:
        return [X[idx] for idx in indices]



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