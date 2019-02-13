##
from typing import Tuple
import entropy_estimators as ee
from math import log
import numpy as np
from numpy.random import multivariate_normal as mn
from numpy.linalg import det
##
def test_entropy():
    np.random.seed(12345)
    results = [[ee.entropy(np.random.rand(200, 2), k=j) for j in range(1, 6)] for _ in range(200)]
    assert(np.all(np.diff(np.mean(results, axis=0)) > 0))

def test_mutual_info():
    r = 0.6
    sqrtm_r = (1 - np.sqrt(1 - r * r)) / r
    np.random.seed(12345)
    res = list()
    for _ in range(1000):
        a, b = np.dot([[1, sqrtm_r], [sqrtm_r, 1]], np.random.randn(2, 1000))
        res.append(ee.mutual_info(a[:, np.newaxis], b[:, np.newaxis]))
    real_info = - (0.5 * np.log2(1 - r ** 2))
    assert(abs(np.mean(res) - real_info) < 1E-2)

def cmutual(size, mean, cov, sample_no=100, ci=(0.025, 0.975)):
    # type: (int, np.ndarray, np.ndarray, int, Tuple[float, float]) -> Tuple[float, Tuple[float, float]]
    """shuffle cmi"""
    np.random.seed(12345)
    ent = [ee.mutual_info(*(x.reshape(-1, 1) for x in mn(mean, cov, size).T)) for _ in range(sample_no)]
    ent = np.sort(ent)
    return np.mean(ent), (ent[int(ci[0] * sample_no)], ent[int(ci[1] * sample_no)])

def test_cmi():
    """Conditional Mutual Information between Gaussian random variables."""
    means = np.zeros(3)
    mat = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    cov = mat.T.dot(np.diag([3, 1, 1])).dot(mat)
    cm_info = [cmutual(try_no, means, cov) for try_no in [10, 25, 50, 100, 200]]
    xz_sub = (cov[0, 0] * cov[2, 2] - cov[0, 2] * cov[2, 0])
    yz_sub = (cov[1, 1] * cov[2, 2] - cov[1, 2] * cov[2, 1])
    true_ent = 0.5 * log(xz_sub * yz_sub / (det(cov) * cov[2, 2])) / log(2)
    for cm_mean, (cm_lower_ci, cm_upper_ci) in cm_info:
        assert(cm_upper_ci > true_ent > cm_lower_ci)

def mutual(size, mean, cov, sample_no=100, ci=(0.025, 0.975)):
    # type: (int, np.ndarray, np.ndarray, int, Tuple[float, float]) -> Tuple[float, Tuple[float, float]]
    np.random.seed(12345)
    ent = [ee.mutual_info(*(x.reshape(-1, 1) for x in mn(mean, cov, size).T[0: 2, :])) for _ in range(sample_no)]
    ent = np.sort(ent)
    return np.mean(ent), (ent[int(ci[0] * sample_no)], ent[int(ci[1] * sample_no)])

def test_mi_repeats():
    """Mutual Information between Gaussian random variables."""
    means = np.zeros(3)
    mat = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    cov = mat.T.dot(np.diag([3, 1, 1])).dot(mat)
    true_ent = 0.5 * log(cov[0, 0] * cov[1, 1] / det(cov[0: 2, 0: 2])) / log(2)
    m_info = [mutual(try_no, means, cov) for try_no in [10, 25, 50, 100, 200]]
    for m_mean, (m_lower_ci, m_upper_ci) in m_info:
        assert(m_upper_ci > true_ent > m_lower_ci)

def test_kldiv():
    """Test divergence estimator (not symmetric, not required to have same num
    samples in each sample set.
    """
    np.random.seed(12345)
    sample1 = np.random.rand(200, 2)
    # should be 0 for same distribution
    res = [ee.kl_divergence(sample1, np.random.rand(200, 2)) for _ in range(1000)]
    assert(abs(np.mean(res)) < 0.2)
    sample1 += 3
    res2 = [ee.kl_divergence(sample1, np.random.rand(200, 2)) for _ in range(1000)]
    # should be infinite for totally disjoint distributions
    # (but this estimator has an upper bound like log(dist) between disjoint prob. masses)
    assert(abs(np.mean(res2) - 11) < 0.1)
