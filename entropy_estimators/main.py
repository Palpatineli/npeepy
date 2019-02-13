"""Non-parametric Entropy Estimation Toolbox

This package contains Python code implementing several entropy estimation
functions for both discrete and continuous variables.

Written by Greg Ver Steeg

See readme.pdf for documentation
Or go to http://www.isi.edu/~gregv/npeet.html
"""
from typing import Optional, Tuple
from scipy.spatial import cKDTree
from scipy.special import digamma as ψ
from math import log
import numpy as np
import warnings

__all__ = ["entropy", "mutual_info", "mutual_info_mixed", "kl_divergence", "shuffle_test"]

# CONTINUOUS ESTIMATORS

def _format_sample(x):
    # type: (np.ndarray) -> np.ndarray
    x = _jitter(np.asarray(x))
    assert x.ndim < 3, "x can only be 1D or 2D"
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x

def _entropy(x, k=3, base=2):
    # type: (np.ndarray, int, float) -> float
    """The classic K-L k-nearest neighbor continuous entropy estimator.
    Estimates the (differential) entropy of :math:`x \in \mathbb{R}^{d_x}`
    from samples :math:`x^{(i)}, i = 1, ..., N`. Differential entropy,
    unlike discrete entropy, can be negative due to close neighbors having
    negative distance.

    Args:
        ndarray[float] x: a list of vectors,
            e.g. x = [[1.3], [3.7], [5.1], [2.4]]
            if x is a one-dimensional scalar and we have four samples
        int k: use k-th neighbor
        float base: unit of the returned entropy
    Returns:
        float: in bit if base is 2, or nat if base is e
    """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x = _format_sample(x)
    n_elements, n_features = x.shape
    neighbor_distances = _neighbor(x, k)
    const = ψ(n_elements) - ψ(k) + n_features * log(2)
    return (const + n_features * np.log(neighbor_distances).mean()) / log(base)

def entropy(x, y=None, k=3, base=2):
    # type: (np.ndarray, Optional[np.ndarray], int, float) -> float
    """The classic K-L k-nearest neighbor continuous entropy estimator.
    Estimates the (differential) entropy of :math:`x \in \mathbb{R}^{d_x}`
    from samples :math:`x^{(i)}, i = 1, ..., N`. Differential entropy,
    unlike discrete entropy, can be negative due to close neighbors having
    negative distance. If y is provided then it gives entropy of x conditioned on y.

    Args:
        ndarray[vector] x, y: a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
            if x is a one-dimensional scalar and we have four samples
        int k: use k-th neighbor
        float base: unit of the returned entropy
    Returns:
        float: in bit if base is 2, or nat if base is e
    """
    if y is None:
        return _entropy(x, k=k, base=base)
    else:
        return _entropy(np.c_[x, y], k=k, base=base) - _entropy(y, k=k, base=base)

def mutual_info(x, y, z=None, k=3, base=2):
    # type: (np.ndarray, np.ndarray, Optional[np.ndarray], int, float) -> float
    """ Estimate the mutual information between :math:`x \in \mathbb{R}^{d_x}`
    and :math:`y \in \mathbb{R}^{d_y}` from samples import
    :math:`x^{(i)}, y^{(i)}, i = 1, ..., N`, conditioned on z if z is not None.

    Args:
        ndarray[vector] x, y: a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
            if x is a one-dimensional scalar and we have four samples
        ndarray[vector] z (, optional): a list of vectors with same length as x and y
        int k: use k-th neighbor
        float base: unit of entropy
    Returns:
        float: mutual information
    """
    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x, y = _format_sample(x), _format_sample(y)
    points = np.c_[x, y] if z is None else np.c_[x, y, z]
    distances = _neighbor(points, k)
    if z is None:
        return (ψ(k) + ψ(len(x)) - _ψ_avg(x, distances) - _ψ_avg(y, distances)) / log(base)
    else:
        return (_ψ_avg(z, distances) + ψ(k)
                - _ψ_avg(np.c_[x, z], distances) - _ψ_avg(np.c_[y, z], distances)) / log(base)

def kl_divergence(x, x_prime, k=3, base=2):
    # type: (np.ndarray, np.ndarray, int, float) -> float
    """Estimate the KL divergence between two distributions
    :math:`p(x)` and :math:`q(x)` from samples x, drawn from :math:`p(x)` and samples
    :math:`x'` drawn from :math:`q(x)`. The number of samples do no have to be the same.
    KL divergence is not symmetric.

    Args:
        np.ndarray[vector] x, x_prime: list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
            if x is a one-dimensional scalar and we have four samples
        int k: use k-th neighbor
        float base: unit of entropy
    Returns:
        float: divergence
    """
    assert k < min(len(x), len(x_prime)), "Set k smaller than num. samples - 1"
    assert len(x[0]) == len(x_prime[0]), "Two distributions must have same dim."
    n, d, m = len(x), len(x[0]), len(x_prime)
    const = log(m) - log(n - 1)
    nn, nn_prime = _neighbor(x, k), _neighbor(x_prime, k - 1)
    return (const + d * (np.log(nn_prime).mean() - np.log(nn).mean())) / log(base)

def _entropy_discrete(x, base=2):
    # type: (np.ndarray, float) -> float
    """Estimates entropy given a list of samples of discrete variable x.
    where :math:`\hat{p} = \\frac{count}{total\:number}`

    Args:
        np.array[vector] sx: a list of samples
        float base: unit of entropy
    Returns:
        float: entropy
    """
    unique, count = np.unique(x, return_counts=True, axis=0)
    prob = count / len(x)
    return np.sum(prob * np.log(1. / prob)) / log(base)

def entropy_discrete(x, y=None, base=2):
    # type: (np.ndarray, Optional[np.ndarray], float) -> float
    """ Estimates entropy for samples from discrete variable X conditioned on
    discrete variable Y
    Args:
        ndarray[obj] x, y: list of samples which can be any hashable object,
            if y is not None then give entropy conditioned on y
    Returns:
        float: conditional entropy
    """
    if y is None:
        return _entropy_discrete(x, base=base)
    else:
        return _entropy_discrete(np.c_[x, y], base) - _entropy_discrete(y, base)

def mutual_info_mixed(x, y, k=3, base=2, warning=True):
    # type: (np.ndarray, np.ndarray, int, float, bool) -> float
    """Estimates the mutual information between a continuous variable :math:`x \in \mathbb{R}^{d_x}`
    and a discrete variable y. Note that mutual information is symmetric, but you must pass the
    continuous variable first.

    Args:
        ndarray[vector] x: list of samples from continuous random variable X, ndarray of vector
        ndarray[vector] y: list of samples from discrete random variable Y, ndarray of vector
        int k: k-th neighbor
        bool warning: provide warning for insufficient data
    Returns:
        float: mutual information
    """
    assert len(x) == len(y), "Arrays should have same length"
    entropy_x = _entropy(x, k, base=base)
    y_unique, y_count, y_index = np.unique(y, return_counts=True, return_inverse=True, axis=0)
    if warning:
        insufficient = np.flatnonzero(y_count < k + 2)
        if len(insufficient) > 0:
            warnings.warn("Warning: y=[{yval}] has insufficient data, "
                          "where we assume maximal entropy.".format(
                              ", ".join([str(a) for a in y_unique[insufficient]])))
    H_x_y = np.array([(_entropy(x[y_index == idx], k=k, base=base) if count > k else entropy_x)
                      for idx, count in enumerate(y_count)])
    return abs(entropy_x - H_x_y * y_count / len(y))  # units already applied

def _jitter(x, intensity=1e-10):
    # type: (np.ndarray, float) -> np.ndarray
    """Small noise to break degeneracy, as points with same coordinates screws nearest neighbor.
    Noise distribution doesn't really matter as it's supposed to be extremely small."""
    return x + intensity * np.random.random_sample(x.shape)

def _neighbor(x, k):
    # type: (np.ndarray, int) -> np.ndarray
    """Get the k-th neighbor of a list of vectors.

    Args:
        ndarray[vector] x: a 2d array [n x m] with n samples and samples are m-dimensional
        int k: k-th neighbor
    Returns:
        ndarray: 1D array for distance between each sample and its k-th nearest neighbor
    """
    # n_jobs = -1: all processes used
    return cKDTree(x).query(x, k=k + 1, p=np.inf, n_jobs=-1)[0][:, k]

def _ψ_avg(x, distances):
    # type: (np.ndarray, np.ndarray) -> float
    """Find number of neighbors in some radius in the marginal space.

    Args:
        ndarray[vector] x: a 2d array [n x m] with n samples and samples are m-dimensional
        ndarray[float] distances: a 1d array [n] with distances to k-th neighbor for each of
            the n samples.
    Returns:
        :math:`E_{<ψ(n_x)>}`
    """
    tree = cKDTree(x)
    # not including the boundary point is equivalent to +1 to n_x. as center point is included
    return np.mean([ψ(len(tree.query_ball_point(a, dist, p=np.inf))) for a, dist in zip(x, distances - 1E-15)])

# TESTS
def shuffle_test(measure,  # Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]
                 x,  # np.ndarray
                 y,  # np.ndarray
                 z=None,  # Optional[np.ndarray]
                 ns=200,  # int
                 ci=0.95,  # floatt
                 **kwargs):
    # type: (...) -> Tuple[float, Tuple[float, float]]
    """Shuffle the x's so that they are uncorrelated with y,
    then estimates whichever information measure you specify with 'measure'.
    e.g., mutual information with mi would return the average mutual information
    (which should be near zero, because of the shuffling) along with the confidence
    interval. This gives a good sense of numerical error and, particular, if your
    measured correlations are stronger than would occur by chance.

    Args:
        (ndarray,ndarray,Optiona[ndarray])->float measure: the function
        ndarray x, y: x and y for measure
        ndarray z: if measure takes z, then z is given here
        int ns: number of shuffles
        float ci: two-side confidence interval
        kwargs: other parameters for measure
    Returns:
        (float,(float,float)): average_value, (lower_confidence, upper_confidence)
    """
    x_clone = np.copy(x)  # A copy that we can shuffle
    outputs = []
    for i in range(ns):
        np.random.shuffle(x_clone)
        outputs.append((measure(x_clone, y, z, **kwargs) if z else measure(x_clone, y, **kwargs)))
    outputs.sort()
    return np.mean(outputs), (outputs[int((1. - ci) / 2 * ns)], outputs[int((1. + ci) / 2 * ns)])
