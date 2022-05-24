#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
import sys
import numpy as np
import pandas as pd
import scipy.linalg

# public symbols
__all__ = ['Model']


class Model(object):
    """
    Base class for models
    """

    @abstractmethod
    def sampling(self, lam):
        """
        Abstract method for sampling.
        :param int lam: sample size :math:`\\lambda`
        :return: samples
        """
        pass

    @abstractmethod
    def loglikelihood(self, X):
        """
        Abstract method for log likelihood.
        :param X: samples
        :return: log likelihoods
        """
        pass

    def terminate_condition(self):
        """
        Check terminate condition.
        :return bool: terminate condition is satisfied or not
        """
        return False

    def verbose_display(self):
        """
        Return verbose display string.
        :return str: string for verbose display
        """
        return ''

    def log_header(self):
        """
        Return model log header list.
        :return: header info list for model log
        :rtype string list:
        """
        return []

    def log(self):
        """
        Return model log string list.
        :return: model log string list
        :rtype string list:
        """
        return []


class Gaussian(Model):
    """
    Gaussian distribution parametrized by mean vector :math:`m` and (full) covariance matrix :math:`C`.

    :param int d: dimension
    :param m: mean vector :math:`m` (option, default is numpy.zeros(d))
    :param C: covariance matrix :math:`C` (option, default is numpy.identity(d))
    :param float minimal_eigenval: minimal eigenvalue for terminate condition
    :type m: array_like, shape(d), dtype=float
    :type C: array_like, shape(d, d), dtype=float
    """

    def __init__(self, d, m=None, C=None, minimal_eigenval=1e-30, normalize='None'):
        self.normalize = normalize
        self.d = d
        self.m = m if m is not None else np.zeros(self.d)
        self.C = C if C is not None else np.identity(self.d)
        self.init_C = C
        self.min_eigenval = minimal_eigenval
        

        if len(self.m) != d or self.C.shape != (d, d):
            print("The size of parameters is invalid.")
            print("Dimension: %d, Mean vector: %d, Covariance matrix: %s" % (self.d, len(self.m), self.__C.shape))
            print("at " + self.__class__.__name__ + " class")
            sys.exit(1)

    def _get_C(self):
        return self.__C

    def _set_C(self, C):
        # Dealing with symmetry breaking due to computational errors
        
        if self.normalize == 'det':
            C_ = (C + C.T) / 2
            normalization_coeff = (np.linalg.det(self.init_C) / np.linalg.det(C_)) ** (1 / len(C))
            self.__C = normalization_coeff * C_
        elif self.normalize == 'trace':
            C_ = (C + C.T) / 2
            normalization_coeff = np.trace(self.init_C) / np.trace(C_)
            self.__C = normalization_coeff * C_
        else:
            self.__C = (C + C.T) / 2
        self.__eigen_decomposition()

    C = property(_get_C, _set_C)

    def sampling(self, lam):
        """
        Draw :math:`\\lambda` samples from the Gaussian distribution.

        :param int lam: sample size :math:`\\lambda`
        :return: sampled vectors from :math:`\\mathcal{N}(m, C)` Gaussian distribution
        :rtype: array_like, shape=(lam, d), dtype=float
        """
        return np.random.randn(lam, self.d).dot(self.sqrtC.T) + self.m

    def loglikelihood(self, X):
        """
        Calculate log likelihood.

        :param X: samples
        :type X: array_like, shape=(lam, d), dtype=float
        :return: log likelihoods
        :rtype: array_like, shape=(lam), dtype=float
        """
        Z = np.dot((X - self.m), self.invSqrtC.T)
        return - 0.5 * (self.d * np.log(2. * np.pi) + self.logDetC) - 0.5 * np.linalg.norm(Z, axis=1)**2

    def terminate_condition(self):
        return (np.min(self.eigvals) < self.min_eigenval) | (0 < (np.isinf(self.m).sum() + np.isinf(self.C).sum())) | (0 < (np.isnan(self.m).sum() + np.isnan(self.C).sum() ))

    def verbose_display(self):
        return ' MinEigVal: %e' % (np.min(self.eigvals))

    def log_header(self):
        return ['m%d' % i for i in range(self.d)] + [f'diagC{i}' for i in range(self.d) ] + ['eigval%d' % i for i in range(self.d)] + ['logDetC']

    def log(self):
        return ['%e' % i for i in self.m] + ['%e' % i for i in np.diag(self.C)] + ['%e' % i for i in self.eigvals] + ['%e' % self.logDetC]

    # Private method
    def __eigen_decomposition(self):
        self.eigvals, self.eigvectors = scipy.linalg.eigh(self.C, driver='ev')
        B = self.eigvectors

        if np.min(self.eigvals) > 0.:
            D = np.diag(np.sqrt(self.eigvals))
            self.sqrtC = np.dot(np.dot(B, D), B.T)
            # self.invC = np.dot(np.dot(B, np.diag(np.reciprocal(self.eigvals))), B.T)
            self.invSqrtC = np.dot(np.dot(B, np.diag(np.reciprocal(np.sqrt(self.eigvals)))), B.T)
            self.logDetC = np.log(self.eigvals).sum()
        else:
            print('The minimal eigenvalue becomes negative value!')

class GaussianSigmaACA(Gaussian):
    def __init__(self, d, z_space, m=None, C=None, sigma=1., minimal_eigenval=1e-30, normalize='None'):
        super().__init__(d, m=m, C=C, minimal_eigenval=minimal_eigenval, normalize=normalize)
        self.sigma = sigma
        # discrete variables
        self.zd = len(z_space)
        # parameter for Affine Map (std)
        self.A = np.full(d, 1.)
        # boder
        lim = (z_space[:,1:] + z_space[:,:-1])/2
        # nan to maxima
        df_a = pd.DataFrame(z_space.T)
        df_li = pd.DataFrame(lim.T)
        self.z_space = df_a.fillna(df_a.max()).values.T
        self.z_lim = df_li.fillna(df_li.max()).values.T
        self.z_lim_low = np.concatenate([self.z_lim.min(axis=1).reshape([self.zd,1]), self.z_lim], 1)
        self.z_lim_up = np.concatenate([self.z_lim, self.z_lim.max(axis=1).reshape([self.zd,1])], 1)
        m_z = m[self.d - self.zd:].reshape(([self.zd, 1]))
        # m_z_lim_low ->|  mean vector    |<- m_z_lim_up
        self.m_z_lim_low = (self.z_lim_low * np.where(np.sort(np.concatenate([self.z_lim, m_z], 1))==m_z, 1, 0)).sum(axis=1)
        self.m_z_lim_up = (self.z_lim_up * np.where(np.sort(np.concatenate([self.z_lim, m_z], 1))==m_z, 1, 0)).sum(axis=1)

    def sampling(self, lam):
        return self.sigma * np.random.randn(lam, self.d).dot(self.sqrtC.T) + self.m
    
    def encoding(self, lam, X):
        """
        X.shape = (lam, N_continuous + N_integer) 
        """
        # Affine Mapped Samples
        X = (X - self.m) * self.A + self.m
        num_cont = self.d - self.zd # = N_continuous
        # get variables for discrete
        X_z = X[:,num_cont:]
        # reshape variables for discrete
        X_z_c = X_z.reshape(([lam, self.zd, 1]))
        # encoding
        X_z_enc = (self.z_space * np.where(np.sort(np.concatenate([np.tile(self.z_lim, (lam,1,1)), X_z_c], 2))==X_z_c, 1, 0)).sum(axis=2)
        return np.hstack((X[:,:num_cont], X_z_enc))

    def loglikelihood(self, X):
        Z = np.dot((X - self.m), self.invSqrtC.T) / self.sigma
        return - 0.5 * (self.d * np.log(2. * np.pi) + self.logDetC) - np.log(self.sigma) - 0.5 * np.linalg.norm(Z, axis=1)**2

    def terminate_condition(self):
        return np.logical_or(np.logical_or((self.sigma**2) * np.min(self.eigvals) < self.min_eigenval, (0 < (np.isinf(self.m).sum() + np.isinf(self.C).sum() + np.isinf(self.sigma).sum()))), (0 < (np.isnan(self.m).sum() + np.isnan(self.C).sum() + np.isnan(self.sigma).sum())))

    def verbose_display(self):
        return ' MinEigVal: %e' % ((self.sigma**2) * (np.min(self.eigvals)))

    def log_header(self):
        return super().log_header()

    def log(self):
        return super().log()