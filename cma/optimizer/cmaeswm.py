#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import copy
import scipy.linalg
from scipy.stats import chi2
from scipy.stats import norm

from ..util.weight import CMAWeight, CMAWeightWithNegativeWeights
from ..optimizer.base_optimizer import BaseOptimizer
from ..util.model import GaussianSigmaACA


# public symbols
__all__ = ['CMAESwM']


class CMAParam(object):
    """
    Default parameters for CMA-ES.
    """
    @staticmethod
    def pop_size(dim):
        return 4 + int(np.floor(3 * np.log(dim)))

    @staticmethod
    def mu_eff(lam, weights=None):
        if weights is None and lam < 4:
            weights = CMAWeight(4).w
        if weights is None:
            weights = CMAWeight(lam).w
        w_1 = np.absolute(weights).sum()
        return w_1**2 / weights.dot(weights)

    @staticmethod
    def c_1(dim, mueff):
        return 2.0 / ((dim + 1.3) * (dim + 1.3) + mueff)

    @staticmethod
    def c_mu(dim, mueff, c1=0., alpha_mu=2.):
        return np.minimum(1. - c1, alpha_mu * (mueff - 2. + 1./mueff) / ((dim + 2.)**2 + alpha_mu * mueff / 2.))

    @staticmethod
    def c_c(dim, mueff):
        return (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)

    @staticmethod
    def c_sigma(dim, mueff):
        return (mueff + 2.0) / (dim + mueff + 5.0)

    @staticmethod
    def damping(dim, mueff):
        return 1.0 + 2.0 * np.maximum(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + CMAParam.c_sigma(dim, mueff)

    @staticmethod
    def chi_d(dim):
        return np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))  # ||N(0,I)||


class CMAESwM(BaseOptimizer):
    def __init__(self, d, discrete_space, weight_func, sampler, m=None, C=None, sigma=1., minimal_eigenval=1e-30,
                 lam=None, c_m=1., c_1=None, c_mu=None, c_c=None, c_sigma=None, damping=None,
                 alpha_mu=2., margin=None, restart=None, mean_regenerate_func=None, normalize='None', reset_margin=False, local_restart=None):
        self.model = GaussianSigmaACA(d, m=m, C=C, sigma=sigma, z_space=discrete_space, minimal_eigenval=minimal_eigenval, normalize=normalize)
        self.model_init = GaussianSigmaACA(d, m=m, C=C, sigma=sigma, z_space=discrete_space, minimal_eigenval=minimal_eigenval, normalize=normalize)
        self.weight_func = weight_func
        self.sampler = sampler
        self.lam = lam if lam is not None else CMAParam.pop_size(d)
        self.d = d
        self.zd = len(discrete_space)

        # CMA parameters
        self.alpha_mu = alpha_mu
        self.mu_eff = CMAParam.mu_eff(self.lam)
        self.c_1 = CMAParam.c_1(d, self.mu_eff) if c_1 is None else c_1
        self.c_mu = CMAParam.c_mu(d, self.mu_eff, c1=self.c_1, alpha_mu=alpha_mu) if c_mu is None else c_mu
        self.c_c = CMAParam.c_c(d, self.mu_eff) if c_c is None else c_c
        self.c_sigma = CMAParam.c_sigma(d, self.mu_eff) if c_sigma is None else c_sigma
        self.damping = CMAParam.damping(d, self.mu_eff) if damping is None else damping
        self.chi_d = CMAParam.chi_d(d)
        self.c_m = c_m

        # evolution path
        self.ps = np.zeros(d)
        self.pc = np.zeros(d)
        self.gen_count = 0

        # restarts
        self.max_restart = restart if restart is not None else 9
        self.restart_count = 0
        self.eval_hist = []         # short history of best
        self.best_eval = np.inf     # minimization problem
        self.not_improve_ite = 0
        self.best_evals = []
        # restarts (parameters)
        self.Tolfun = 1e-12
        self.TolX = 1e-12
        self.maxcond = 1e14
        self.mean_regenerate_func = mean_regenerate_func
        self.reset_margin = reset_margin
        self.local_restart = local_restart
        self.best_candidate = np.zeros(self.d)

        # margin parameter (alpha in the paper)
        self.margin = margin if margin is not None else 1 / (d * lam)

    def sampling_model(self):
        return self.model

    def update(self, X, evals):
        self.gen_count += 1

        # eval history
        best_eval_t = np.min(evals)
        self.eval_hist.insert(0, best_eval_t)
        if len(self.eval_hist) > 10 + 30 * self.d / self.lam:
            self.eval_hist.pop()
        if self.best_eval > best_eval_t:
            self.best_eval = best_eval_t
            self.best_candidate = X[np.argmin(evals)]
            self.not_improve_ite = 0
        else:
            self.not_improve_ite += 1

        # mean vector
        weights = self.weight_func(evals)
        Y = (X - self.model.m) / self.model.sigma
        weights_for_mean_update = np.zeros_like(weights)
        weights_for_mean_update[weights > 0] = weights[weights > 0]
        WYT = weights_for_mean_update * Y.T
        m_diff = self.model.sigma * WYT.sum(axis=1)
        
        # covariance
        weights_for_cov_update = np.zeros_like(weights)
        if np.any(weights < 0):
            weights_for_cov_update[weights > 0] = weights[weights > 0]
            weights_for_cov_update[weights < 0] = weights[weights < 0] * self.d / (scipy.linalg.norm(np.dot(self.model.invSqrtC, Y[weights < 0].T), axis=0) ** 2 + 1e-10)
        else:
            weights_for_cov_update = weights
        WYT = weights_for_cov_update * Y.T
        C_rank_mu = np.dot(WYT, Y) - weights.sum() * self.model.C

        hsig = 1.
        # compute evolution path and CSA (Original CMA-ES version) 
        if self.c_1 != 0. or self.damping != np.inf:
            # evolution path
            self.ps = (1.0 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * np.dot(self.model.invSqrtC, self.c_m * m_diff / self.model.sigma)
            hsig = 1. if scipy.linalg.norm(self.ps) / (np.sqrt(1. - (1. - self.c_sigma) ** (2 * self.gen_count + 1))) < (1.4 + 2. / (self.model.d + 1.)) * self.chi_d else 0.
            self.pc = (1. - self.c_c) * self.pc + hsig * np.sqrt(self.c_c * (2. - self.c_c) * self.mu_eff) * self.c_m * m_diff / self.model.sigma
        if self.damping != np.inf:
            # CSA
            self.model.sigma = self.model.sigma * np.exp(self.c_sigma / (1. * self.damping) * (scipy.linalg.norm(self.ps) / self.chi_d - 1.))
        
        # mean vector update
        self.model.m = self.model.m + self.c_m * m_diff
        # covariance matrix update
        self.model.C = self.model.C + (1.-hsig)*self.c_1*self.c_c*(2.-self.c_c)*self.model.C + self.c_1 * (np.outer(self.pc, self.pc) - self.model.C) + self.c_mu * C_rank_mu
        
        # margin correction (if self.margin = 0, this behaves as CMA-ES)
        if self.margin > 0.:
            num_cont = self.model.d - self.model.zd # = N_continuous
            updated_m_integer = self.model.m[num_cont:, np.newaxis]
            # m_z_lim_low ->|  mean vector    |<- m_z_lim_up
            self.z_lim_low = np.concatenate([self.model.z_lim.min(axis=1).reshape([self.model.zd,1]), self.model.z_lim], 1)
            self.z_lim_up = np.concatenate([self.model.z_lim, self.model.z_lim.max(axis=1).reshape([self.model.zd,1])], 1)
            self.m_z_lim_low = (self.z_lim_low * np.where(np.sort(np.concatenate([self.model.z_lim, updated_m_integer], 1))==updated_m_integer, 1, 0)).sum(axis=1)
            self.m_z_lim_up = (self.z_lim_up * np.where(np.sort(np.concatenate([self.model.z_lim, updated_m_integer], 1))==updated_m_integer, 1, 0)).sum(axis=1)

            # calculate probability low_cdf := Pr(X <= m_z_lim_low) and up_cdf := Pr(m_z_lim_up < X)
            sig_z_sq_Cdiag = self.model.sigma * self.model.A * np.sqrt(np.diag(self.model.C))
            z_scale = sig_z_sq_Cdiag[num_cont:]
            updated_m_integer = updated_m_integer.flatten()
            low_cdf = norm.cdf(self.m_z_lim_low, loc = updated_m_integer, scale = z_scale)
            up_cdf = 1. - norm.cdf(self.m_z_lim_up, loc = updated_m_integer, scale = z_scale)
            mid_cdf = 1. - (low_cdf + up_cdf)
            # edge case
            edge_mask = (np.maximum(low_cdf, up_cdf) > 0.5)
            # otherwise
            side_mask = (np.maximum(low_cdf, up_cdf) <= 0.5)
       
            if np.any(edge_mask):
                # modify mask (modify or not)
                modify_mask = (np.minimum(low_cdf, up_cdf) < self.margin)
                # modify sign
                modify_sign = np.sign(self.model.m[num_cont:] - self.m_z_lim_up)
                # distance from m_z_lim_up
                dist = self.model.sigma * self.model.A[num_cont:] * np.sqrt(chi2.ppf(q = 1.-2.*self.margin, df = 1) * np.diag(self.model.C)[num_cont:])
                # modify mean vector
                self.model.m[num_cont:] = self.model.m[num_cont:] + modify_mask * edge_mask * (self.m_z_lim_up + modify_sign * dist - self.model.m[num_cont:])

            # correct probability
            low_cdf = np.maximum(low_cdf, self.margin/2.)
            up_cdf = np.maximum(up_cdf, self.margin/2.)
            modified_low_cdf = low_cdf + (1. - low_cdf - up_cdf - mid_cdf) * (low_cdf - self.margin / 2) / (low_cdf + mid_cdf + up_cdf - 3. * self.margin / 2)
            modified_up_cdf = up_cdf + (1. - low_cdf - up_cdf - mid_cdf) * (up_cdf - self.margin / 2) / (low_cdf + mid_cdf + up_cdf - 3. * self.margin / 2)
            modified_low_cdf = np.clip(modified_low_cdf, 1e-10, 0.5 - 1e-10)
            modified_up_cdf = np.clip(modified_up_cdf, 1e-10, 0.5 - 1e-10)
        
            # modify mean vector and A (with sigma and C fixed)
            chi_low_sq = np.sqrt(chi2.ppf(q = 1.-2*modified_low_cdf, df = 1))
            chi_up_sq = np.sqrt(chi2.ppf(q = 1.-2*modified_up_cdf, df = 1))
            C_diag_sq = np.sqrt(np.diag(self.model.C))[num_cont:]

            # simultaneous equations
            # (updated_m_integer) - self.m_z_lim_low = chi_low_sq * self.model.sigma * (self.model.A) * C_diag_sq
            # self.m_z_lim_up - (updated_m_integer) = chi_up_sq * self.model.sigma * (self.model.A) * C_diag_sq
            self.model.A[num_cont:] = self.model.A[num_cont:] + side_mask * ( (self.m_z_lim_up - self.m_z_lim_low) / ((chi_low_sq + chi_up_sq) * self.model.sigma * C_diag_sq) - self.model.A[num_cont:] )
            self.model.m[num_cont:] = self.model.m[num_cont:] + side_mask * ( (self.m_z_lim_low * chi_up_sq + self.m_z_lim_up * chi_low_sq) / (chi_low_sq + chi_up_sq) - self.model.m[num_cont:] )

        # end of margin correction

        # restart_check
        flag_count = self.restart_count < self.max_restart
        flag_not_improve = self.not_improve_ite > int(100 + 100 * (self.d**1.5) / self.lam)  # no improvement over int(100 + 100 * N**1.5 / popsize) iterations
        flag_min_std_pc = max(max(self.model.sigma * np.sqrt(np.diag(self.model.C))), max((self.model.sigma) * self.pc)) < self.TolX
        flag_equal_hist = (len(self.eval_hist) >= 10 + 30 * self.d / self.lam) and (max(self.eval_hist) - min(self.eval_hist) < self.Tolfun)
        tmDim = self.gen_count % self.d
        D = np.sort(np.sqrt(np.diag(self.model.C)))[::-1]
        flag_noeffectaxis = all(self.model.m == self.model.m + 0.1 * self.model.sigma * D[tmDim] * self.model.eigvectors[:,tmDim])
        flag_noeffectcoord = any(self.model.m == self.model.m + 0.2 * self.model.sigma * np.sqrt(np.diag(self.model.C)))
        flag_conditioncov = np.linalg.cond(self.model.C) > self.maxcond
        if flag_count and (flag_not_improve or flag_min_std_pc or flag_equal_hist or flag_noeffectaxis or flag_noeffectcoord or flag_conditioncov):
            # IPOP restart
            self.sampler.lam = 2 * self.sampler.lam
            self.lam = 2 * self.lam
            # reinit
            if isinstance(self.weight_func, CMAWeight):
                self.weight_func = CMAWeight(self.lam, min_problem=self.sampler.f.minimization_problem)
            else:
                self.weight_func = CMAWeightWithNegativeWeights(self.lam, self.d, min_problem=self.sampler.f.minimization_problem)
            # CMA parameters
            self.mu_eff = CMAParam.mu_eff(self.lam)
            self.c_1 = CMAParam.c_1(self.d, self.mu_eff)
            self.c_mu = CMAParam.c_mu(self.d, self.mu_eff, c1=self.c_1, alpha_mu=self.alpha_mu)
            self.c_c = CMAParam.c_c(self.d, self.mu_eff)
            self.c_sigma = CMAParam.c_sigma(self.d, self.mu_eff)
            self.damping = CMAParam.damping(self.d, self.mu_eff)            
            
            self.model = copy.copy(self.model_init)
            if self.reset_margin:
                self.margin = 1. / (self.lam * self.d)
            if self.mean_regenerate_func is not None:
                self.model.m = self.mean_regenerate_func()
            if self.local_restart == 'integer':
                self.model.m[num_cont:] = self.model.encoding(1, self.best_candidate.reshape(1, -1))[0, num_cont:]
            elif self.local_restart == 'all':
                self.model.m = self.model.encoding(1, self.best_candidate.reshape(1, -1))[0]
            self.ps = np.zeros(self.d)
            self.pc = np.zeros(self.d)
            self.model.A = np.full(self.d, 1.)
            self.best_evals.append(self.best_eval)
            self.best_eval = np.inf
            self.not_improve_ite = 0
            self.restart_count += 1

    def terminate_condition(self):
        return self.model.terminate_condition()

    def verbose_display(self):
        return self.model.verbose_display() + ' restart : ' + str(self.restart_count) + ' / ' + str(self.max_restart) if self.max_restart >= 1 else self.model.verbose_display()

    def log_header(self):
        return self.model.log_header() + ['sigma'] + ['A%d' % i for i in range(self.d)]

    def log(self):
        return self.model.log() + ['%e' % self.model.sigma] + ['%e' % i for i in self.model.A]