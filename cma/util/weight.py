#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from scipy.stats import rankdata

from ..optimizer import cmaeswm as cma


__all__ = ['CMANonIncFunc', 'SelectionNonIncFunc', 'QuantileBasedWeight', 'CMAWeight']


class CMANonIncFunc(object):
    def __init__(self):
        pass

    @staticmethod
    def calc_weight(q):
        q[q < 1e-300] = 1e-300
        w = - 2. * np.log(2. * q)
        w[q > 0.5] = 0.
        return w

    @staticmethod
    def tie_component(q):
        w = np.zeros_like(q)
        mask = (q != 0)
        w[mask] = np.minimum(q[mask], 0.5) * (CMANonIncFunc.calc_weight(q[mask]) + 2.)
        return w

    def __call__(self, q_plus, q_minus=None):
        if q_minus is None:
            return self.calc_weight(q_plus)

        q_diff = q_plus - q_minus
        idx = (q_diff != 0)

        weights = np.zeros_like(q_plus)
        weights[~idx] = CMANonIncFunc.calc_weight(q_plus[~idx])
        weights[idx] = (CMANonIncFunc.tie_component(q_plus[idx]) - CMANonIncFunc.tie_component(q_minus[idx])) / q_diff[idx]
        return weights


class SelectionNonIncFunc(object):
    def __init__(self, threshold=0.25, negative_weight=False):
        self.threshold = threshold
        self.negative_weight = negative_weight

        if self.negative_weight:
            self.w_val = 1. / (2 * self.threshold)
            self.threshold_negative = 1. - self.threshold
            self.negative_w_val = - 1. / (2 * self.threshold)
        else:
            self.w_val = 1. / (self.threshold)

        if self.negative_weight and self.threshold > 0.5:
            print("The threshold is invalid.")
            print("It should be satisfied \"threshold <= 0.5\"")
            print("at " + self.__class__.__name__ + " class")
            sys.exit(1)

    def calc_q_plus_w(self, q_plus):
        w = np.zeros_like(q_plus)
        w[q_plus <= self.threshold] = self.w_val
        if self.negative_weight:
            w[q_plus > self.threshold_negative] = self.negative_w_val
        return w

    def __call__(self, q_plus, q_minus=None):
        if q_minus is None:
            return self.calc_q_plus_w(q_plus)

        q_diff = q_plus - q_minus
        if self.negative_weight:
            w = self.w_val * (np.minimum(q_plus, self.threshold) - np.minimum(q_minus, self.threshold)) \
                + self.negative_w_val * (np.maximum(q_plus, self.threshold_negative) - np.maximum(q_minus, self.threshold_negative))
        else:
            w = self.w_val * (np.minimum(q_plus, self.threshold) - np.minimum(q_minus, self.threshold))

        idx = (q_diff != 0)
        w[idx] = w[idx] / q_diff[idx]
        w[~idx] = self.calc_q_plus_w(q_plus[~idx])
        return w


class QuantileBasedWeight(object):
    def __init__(self, non_inc_f, tie_case=False, normalization=False, min_problem=True):
        self.min_problem = min_problem
        self.non_inc_f = non_inc_f
        self.tie_case = tie_case
        self.normalization = normalization

    def __call__(self, evals, likelihood_ratio=None):
        pop_size = evals.size
        if likelihood_ratio is None:
            likelihood_ratio = np.ones(pop_size)

        # Quantile estimation
        evals = evals if self.min_problem else -evals
        sort_idx = np.argsort(evals)
        cum_plus = np.cumsum(likelihood_ratio[sort_idx])
        rank_max = (rankdata(evals, method='max') - 1).astype(np.int)
        q_plus = cum_plus[rank_max] / pop_size
        if self.tie_case:
            cum_minus = cum_plus - likelihood_ratio[sort_idx]
            rank_min = (rankdata(evals, method='min') - 1).astype(np.int)
            q_minus = cum_minus[rank_min] / pop_size
        else:
            q_minus = None

        # Non-increasing transformation
        w = self.non_inc_f(q_plus, q_minus) / pop_size

        # Normalization to be summed up the absolute vales to 1
        if self.normalization and np.abs(w).sum() != 0:
            w = w / np.abs(w).sum()

        return w


class CMAWeight(object):
    def __init__(self, lam, dim=None, min_problem=True):
        self.lam = lam
        self.dim = dim
        self.min_problem = min_problem
        self.w = np.maximum(np.log((self.lam + 1.)/2.) - np.log(np.arange(self.lam)+1.), np.zeros(self.lam))
        self.w = self.w / self.w.sum() if self.w.sum() != 0 else self.w
        self.weights = np.zeros_like(self.w)

    def __call__(self, evals):
        evals = evals if self.min_problem else -evals
        index = np.argsort(evals)
        self.weights[index] = self.w

        # tie case check
        unique_val, count = np.unique(evals, return_counts=True)
        if len(evals) == len(unique_val):
            return self.weights

        # tie case: averaging
        for u_val in unique_val[count > 1]:
            duplicate_index = np.where(evals == u_val)
            self.weights[duplicate_index] = self.weights[duplicate_index].mean()
        return self.weights

    
class CMAWeightWithNegativeWeights(object):
    def __init__(self, lam, dim, min_problem=True):
        self.lam = lam
        self.dim = dim
        self.min_problem = min_problem
        self.w_prime = np.log((self.lam + 1.)/2.) - np.log(np.arange(self.lam)+1.)
        self.mu_eff = np.sum(self.w_prime[self.w_prime > 0]) ** 2 / np.sum(self.w_prime[self.w_prime > 0] ** 2)
        self.mu_eff_neg = np.sum(self.w_prime[self.w_prime <= 0]) ** 2 / np.sum(self.w_prime[self.w_prime <= 0] ** 2)
        self.c_1 = cma.CMAParam().c_1(self.dim, self.mu_eff)
        self.c_mu = cma.CMAParam().c_mu(self.dim, self.mu_eff, self.c_1)
        self.alpha_mu_neg = 1 + self.c_1 / self.c_mu
        self.alpha_mu_eff_neg = 1 + (2 * self.mu_eff_neg / (self.mu_eff + 2))
        self.alpha_pos_def_neg = (1 - self.c_1 - self.c_mu) / (self.dim * self.c_mu) 
        
        self.w = np.zeros_like(self.w_prime)
        self.w[self.w_prime >= 0] = self.w_prime[self.w_prime >= 0] / np.abs(self.w_prime[self.w_prime >= 0]).sum()
        self.w[self.w_prime < 0] = self.w_prime[self.w_prime < 0] \
                                    * min(self.alpha_mu_neg,
                                          self.alpha_mu_eff_neg, 
                                          self.alpha_pos_def_neg) \
                                    / np.abs(self.w_prime[self.w_prime < 0]).sum()
        self.weights = np.zeros_like(self.w_prime)

    def __call__(self, evals):
        evals = evals if self.min_problem else -evals
        index = np.argsort(evals)
        self.weights[index] = self.w

        # tie case check
        unique_val, count = np.unique(evals, return_counts=True)
        if len(evals) == len(unique_val):
            return self.weights

        # tie case: averaging
        for u_val in unique_val[count > 1]:
            duplicate_index = np.where(evals == u_val)
            self.weights[duplicate_index] = self.weights[duplicate_index].mean()
        return self.weights