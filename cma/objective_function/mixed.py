#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..objective_function.base import *
import numpy as np


# public symbols
__all__ = ['initial_setting_for_gaussian', 'SphereOneMax', 'SphereLeadingOnes', 'LeadingOnes', 'SphereInt', 'EllipsoidOneMax', 'EllipsoidLeadingOnes', 'EllipsoidInt']


def initial_setting_for_gaussian(func_instance, random=True):
    """
    Return random initial vector within the range or constant initial vector.

    :type func_instance: object
    :type random: bool
    :return: initial mean vector
    :rtype: array_like, shape=(d), dtype=float
    :return: initial sigma
    :rtype: float
    """
    a, b = 1., 3.
    # mean vector:
    # continuous and integer: sample from uniform distribution [a, b]
    # binary: 0.5
    # sigma: (b - a)/2
    return np.hstack((((b - a) * np.random.rand(func_instance.d - func_instance.bid) + a), (np.full(func_instance.bid, 0.5)))) if random else np.hstack(((b + a) / 2. * np.ones(func_instance.d - func_instance.bid), (np.full(func_instance.bid, 0.5)))), (b - a) / 2.


class SphereOneMax(ObjectiveFunction):
    """
    Sphere function + r * OneMax : :math:`f(x) = \\sum_{i=1}^{d-bid} x_i^2 + r * \\sum_{i=1}^bid x_i`
    """
    minimization_problem = True

    def __init__(self, d, ind, bid, ratio=None, target_eval=1e-10, max_eval=1e4):
        super(SphereOneMax, self).__init__(target_eval, max_eval)
        if ratio is None:
            ratio = 1.
        self.d = d
        self.ind = 0
        self.bid = bid
        self.r = ratio

    def __call__(self, X):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        self.eval_count += len(X)
        S = np.array(X)[:,:(self.d - self.bid)]
        B = ( 0. < np.array(X)[:,(self.d - self.bid):] )
        evals = (S**2).sum(axis=1) + self.r * (self.bid - B.sum(axis=1))
        self._update_best_eval(evals)
        return evals


class SphereLeadingOnes(ObjectiveFunction):
    """
    Sphere function + r * LeadingOnes: :math:`f(x) = \\sum_{i=1}^{d-bid} x_i^2 + r * \\sum_{i=1}^bid \\prod_{j=1}^i x_j`
    """
    minimization_problem = True

    def __init__(self, d, ind, bid, ratio=None, target_eval=1e-10, max_eval=1e4):
        super(SphereLeadingOnes, self).__init__(target_eval, max_eval)
        if ratio is None:
            ratio = 1.
        self.d = d
        self.ind = 0
        self.bid = bid
        self.r = ratio

    def __call__(self, X):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        self.eval_count += len(X)
        S = np.array(X)[:,:(self.d - self.bid)]
        B = ( 0. < np.array(X)[:,(self.d - self.bid):] )
        evals = (S**2).sum(axis=1) + self.r * (self.bid - (B.argmin(axis=1) + B.prod(axis=1) * (self.bid)))
        self._update_best_eval(evals)
        return evals

class SphereInt(ObjectiveFunction):
    """
    Sphere function(rounding(x))
    """
    minimization_problem = True

    def __init__(self, d, ind, bid, ratio=None, target_eval=1e-10, max_eval=1e4):
        super(SphereInt, self).__init__(target_eval, max_eval)
        if ratio is None:
            ratio = 1.
        self.d = d
        self.ind = ind
        self.bid = bid
        self.r = ratio

    def __call__(self, X):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        self.eval_count += len(X)
        SI = np.array(X)
        SI[:,(self.d - self.ind):] = np.round(SI[:,(self.d - self.ind):])
        evals = (SI**2).sum(axis=1)
        #print(SI[:,(self.d - self.ind):])
        #print((np.round(SI[:,(self.d - self.ind):]) != 0).sum(axis = 0))
        self._update_best_eval(evals)
        return evals

class EllipsoidOneMax(ObjectiveFunction):
    """
    Ellipsoid function + r * OneMax: :math:`f(x) = \\sum_{i=1}^{d-bid} (1000^{\\frac{j-1}{d-bid-1}} x_i)^2 + r * \\sum_{i=1}^bid x_i`
    """
    minimization_problem = True

    def __init__(self, d, ind, bid, ratio=None, target_eval=1e-10, max_eval=1e4):
        super(EllipsoidOneMax, self).__init__(target_eval, max_eval)
        if ratio is None:
            ratio = 1.
        self.d = d
        self.ind = 0
        self.bid = bid
        self.r = ratio
        self.coefficient = 1000 ** (np.arange(self.d - self.bid) / float(self.d - self.bid - 1))

    def __call__(self, X):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        self.eval_count += len(X)
        S = np.array(X)[:,:(self.d - self.bid)]
        tmp = S * self.coefficient
        B = ( 0. < np.array(X)[:,(self.d - self.bid):] )
        evals = (tmp**2).sum(axis=1) + self.r * (self.bid - B.sum(axis=1))
        self._update_best_eval(evals)
        return evals


class EllipsoidLeadingOnes(ObjectiveFunction):
    """
    Ellipsoid function + r * LeadingOnes: :math:`f(x) = \\sum_{i=1}^{d-bid} (1000^{\\frac{j-1}{d-bid-1}} x_i)^2 + r * \\sum_{i=1}^bid \\prod_{j=1}^i x_j`
    """
    minimization_problem = True

    def __init__(self, d, ind, bid, ratio=None, target_eval=1e-10, max_eval=1e4):
        super(EllipsoidLeadingOnes, self).__init__(target_eval, max_eval)
        if ratio is None:
            ratio = 1.
        self.d = d
        self.ind = 0
        self.bid = bid
        self.r = ratio
        self.coefficient = 1000 ** (np.arange(self.d - self.bid) / float(self.d - self.bid - 1))

    def __call__(self, X):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        self.eval_count += len(X)
        S = np.array(X)[:,:(self.d - self.bid)]
        tmp = S * self.coefficient
        B = ( 0. < np.array(X)[:,(self.d - self.bid):] )
        evals = (tmp**2).sum(axis=1) + self.r * (self.bid - (B.argmin(axis=1) + B.prod(axis=1) * (self.bid)))
        self._update_best_eval(evals)
        return evals


class EllipsoidInt(ObjectiveFunction):
    """
    Ellipsoid function(rounding(x))
    """
    minimization_problem = True

    def __init__(self, d, ind, bid, ratio=None, target_eval=1e-10, max_eval=1e4):
        super(EllipsoidInt, self).__init__(target_eval, max_eval)
        if ratio is None:
            ratio = 1.
        self.d = d
        self.ind = ind
        self.bid = bid
        self.r = ratio
        self.coefficient = 1000 ** (np.arange(d) / float(d - 1))

    def __call__(self, X):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        self.eval_count += len(X)
        SI = np.array(X)
        SI[:,(self.d - self.ind):] = np.round(SI[:,(self.d - self.ind):])
        tmp = SI * self.coefficient
        evals = (tmp**2).sum(axis=1)
        self._update_best_eval(evals)
        return evals

class twoDRosenbrock(ObjectiveFunction):
    """
    Rosenbrock function: :math:`f(x) = (1-x_0)^2 + 2(x_1 - x_0^2)^2, x_0 \in Z, x_1 \in R`
    """
    minimization_problem = True

    def __init__(self, d, ind, bid, ratio=None, target_eval=1e-10, max_eval=1e4):
        super(twoDRosenbrock, self).__init__(target_eval, max_eval)
        if ratio is None:
            ratio = 1.
        self.d = 2
        self.ind = 1
        self.bid = 0
        self.r = ratio

    def __call__(self, X):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        self.eval_count += len(X)
        S = np.array(X)[:,0]
        B = np.round(np.array(X)[:,1])
        evals = 2 * (S - B ** 2) ** 2 + (B - 1) ** 2
        self._update_best_eval(evals)
        return evals