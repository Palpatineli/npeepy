"""Non-parametric Entropy Estimation Toolbox

This package contains Python code implementing several entropy estimation
functions for both discrete and continuous variables.

Written by Greg Ver Steeg

See readme.pdf for documentation
Or go to http://www.isi.edu/~gregv/npeet.html
"""
from typing import Optional, Tuple
from scipy.spatial import cKDTree
from scipy.special import digamma
from math import log
import numpy as np
import warnings

__all__ = ["entropy", "centropy", "mi", "cmi", "kldiv", "entropyd", "centropyd", "midd", "shuffle_test"]

# CONTINUOUS ESTIMATORS

def entropy(x, k=3, base=2):
    # type: (np.ndarray, int, float) -> float
    """ The classic K-L k-nearest neighbor continuous entropy estimator.
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
    x = np.asarray(x)
    n_elements, n_features = x.shape
    x = add_noise(x)
    tree = cKDTree(x)
    nn = query_neighbors(tree, x, k)
    const = digamma(n_elements) - digamma(k) + n_features * log(2)
    return (const + n_features * np.log(nn).mean()) / log(base)


def centropy(x, y, k=3, base=2):
    # type: (np.ndarray, np.ndarray, int, float) -> float
    """ The classic K-L k-nearest neighbor continuous entropy estimator for the
    entropy of X **conditioned on Y**.

    Args:
        ndarray[vector] x, y: a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
            if x is a one-dimensional scalar and we have four samples
        int k: use k-th neighbor
        float base: unit of the returned entropy
    Returns:
        float: in bit if base is 2, or nat if base is e
    """
    xy = np.c_[x, y]
    entropy_union_xy = entropy(xy, k=k, base=base)
    entropy_y = entropy(y, k=k, base=base)
    return entropy_union_xy - entropy_y


def tc(xs, k=3, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    entropy_features = [entropy(col, k=k, base=base) for col in xs_columns]
    return np.sum(entropy_features) - entropy(xs, k, base)


def ctc(xs, y, k=3, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    centropy_features = [centropy(col, y, k=k, base=base) for col in xs_columns]
    return np.sum(centropy_features) - centropy(xs, y, k, base)


def corex(xs, ys, k=3, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    cmi_features = [mi(col, ys, k=k, base=base) for col in xs_columns]
    return np.sum(cmi_features) - mi(xs, ys, k=k, base=base)


def mi(x, y, z=None, k=3, base=2):
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
    x, y = add_noise(np.asarray(x)), add_noise(np.asarray(y))
    points = np.hstack([x, y]) if z is None else np.hstack([x, y, z])
    tree = cKDTree(points)  # nearest neighbors in joint space, where p=inf means max-norm
    dvec = query_neighbors(tree, points, k)
    if z is None:
        a, b, c, d = avgdigamma(x, dvec), avgdigamma(y, dvec), digamma(k), digamma(len(x))
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = avgdigamma(xz, dvec), avgdigamma(yz, dvec), avgdigamma(z, dvec), digamma(k)
    return (-a - b + c + d) / log(base)


def cmi(x, y, z, k=3, base=2):
    # type: (np.ndarray, np.ndarray, np.ndarray, int, float) -> float
    """ Estimate the mutual information between :math:`x \in \mathbb{R}^{d_x}`
    and :math:`y \in \mathbb{R}^{d_y}` from samples import
    :math:`x^{(i)}, y^{(i)}, i = 1, ..., N`, conditioned on z.

    Args:
        ndarray[vector] x, y: a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
            if x is a one-dimensional scalar and we have four samples
        ndarray[vector] z: a list of vectors with same length as x and y
        int k: use k-th neighbor
        float base: unit of entropy
    Returns:
        float: mutual information
    """
    return mi(x, y, z=z, k=k, base=base)


def kldiv(x, x_prime, k=3, base=2):
    # type: (np.ndarray, np.ndarray, int, float) -> float
    """ Estimate the KL divergence between two distributions
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
    tree, treep = cKDTree(x), cKDTree(x_prime)
    nn, nnp = query_neighbors(tree, x, k), query_neighbors(treep, x, k - 1)
    return (const + d * (np.log(nnp).mean() - np.log(nn).mean())) / log(base)


# DISCRETE ESTIMATORS
def entropyd(sx, base=2):
    # type: (np.ndarray, float) -> float
    """Estimates entropy given a list of samples of discrete variable x.
    where :math:`\hat{p} = \\frac{count}{total\:number}`

    Args:
        np.array[vector] sx: a list of samples
        float base: unit of entropy
    Returns:
        float: entropy
    """
    unique, count = np.unique(sx, return_counts=True, axis=0)
    proba = count / len(sx)
    return np.sum(proba * np.log(1. / proba)) / log(base)


def midd(x, y, base=2):
    # type: (np.ndarray, np.ndarray, float) -> float
    """Estimates the mutual information between discrete variables x and y

    Args:
        x, y: list of samples which can be any hashable object
    Returns:
        float: mutual information
    """
    assert len(x) == len(y), "Arrays should have same length"
    return entropyd(x, base) - centropyd(x, y, base)


def cmidd(x, y, z, base=2):
    # type: (np.ndarray, np.ndarray, np.ndarray, float) -> float
    """Estimates mutual information between discrete variables X and Y conditioned on Z

    Args:
        x, y, z: list of samples which can be any hashable object
    Returns:
        float: conditional entropy
    """
    assert len(x) == len(y) == len(z), "Arrays should have same length"
    xz = np.c_[x, z]
    yz = np.c_[y, z]
    xyz = np.c_[x, y, z]
    return entropyd(xz, base) + entropyd(yz, base) - entropyd(xyz, base) - entropyd(z, base)


def centropyd(x, y, base=2):
    # type: (np.ndarray, np.ndarray, float) -> float
    """ Estimates entropy for samples from discrete variable X conditioned on
    discrete variable Y
    Args:
        ndarray[obj] x, y: list of samples which can be any hashable object
    Returns:
        float: conditional entropy
    """
    xy = np.c_[x, y]
    return entropyd(xy, base) - entropyd(y, base)


def tcd(xs, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    entropy_features = [entropyd(col, base=base) for col in xs_columns]
    return np.sum(entropy_features) - entropyd(xs, base)


def ctcd(xs, y, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    centropy_features = [centropyd(col, y, base=base) for col in xs_columns]
    return np.sum(centropy_features) - centropyd(xs, y, base)


def corexd(xs, ys, base=2):
    xs_columns = np.expand_dims(xs, axis=0).T
    cmi_features = [midd(col, ys, base=base) for col in xs_columns]
    return np.sum(cmi_features) - midd(xs, ys, base)


# MIXED ESTIMATORS
def micd(x, y, k=3, base=2, warning=True):
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
    entropy_x = entropy(x, k, base)

    y_unique, y_count = np.unique(y, return_counts=True, axis=0)
    y_proba = y_count / len(y)

    entropy_x_given_y = 0.
    for yval, py in zip(y_unique, y_proba):
        x_given_y = x[(y == yval).all(axis=1)]
        if k <= len(x_given_y) - 1:
            entropy_x_given_y += py * entropy(x_given_y, k, base)
        else:
            if warning:
                warnings.warn("Warning, after conditioning, on y={yval} insufficient data. "
                              "Assuming maximal entropy in this case.".format(yval=yval))
            entropy_x_given_y += py * entropy_x
    return abs(entropy_x - entropy_x_given_y)  # units already applied


def midc(x, y, k=3, base=2, warning=True):
    return micd(y, x, k, base, warning)


def centropycd(x, y, k=3, base=2, warning=True):
    return entropy(x, base) - micd(x, y, k, base, warning)


def centropydc(x, y, k=3, base=2, warning=True):
    return centropycd(y, x, k=k, base=base, warning=warning)


def ctcdc(xs, y, k=3, base=2, warning=True):
    xs_columns = np.expand_dims(xs, axis=0).T
    centropy_features = [centropydc(col, y, k=k, base=base, warning=warning) for col in xs_columns]
    return np.sum(centropy_features) - centropydc(xs, y, k, base, warning)


def ctccd(xs, y, k=3, base=2, warning=True):
    return ctcdc(y, xs, k=k, base=base, warning=warning)


def corexcd(xs, ys, k=3, base=2, warning=True):
    return corexdc(ys, xs, k=k, base=base, warning=warning)


def corexdc(xs, ys, k=3, base=2, warning=True):
    return tcd(xs, base) - ctcdc(xs, ys, k, base, warning)


# UTILITY FUNCTIONS

def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)


def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1, p=float('inf'), n_jobs=-1)[0][:, k]


def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    n_elements = len(points)
    tree = cKDTree(points)
    avg = 0.
    dvec = dvec - 1e-15
    for point, dist in zip(points, dvec):
        # subtlety, we don't include the boundary point,
        # but we are implicitly adding 1 to kraskov def bc center point is included
        num_points = len(tree.query_ball_point(point, dist, p=float('inf')))
        avg += digamma(num_points) / n_elements
    return avg


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


if __name__ == "__main__":
    print("MI between two independent continuous random variables X and Y:")
    print(mi(np.random.rand(1000, 10), np.random.rand(1000, 3), base=2))
