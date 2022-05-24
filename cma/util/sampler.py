#!/usr/bin/env python
# -*- coding: utf-8 -*-

# public symbols
__all__ = ['Sampler']


class Sampler(object):
    def __init__(self, f, lam):
        self.f = f
        self.lam = lam

    def __call__(self, model):
        X = model.sampling(self.lam)
        X_enc = model.encoding(self.lam, X)
        evals = self.f(X_enc)
        return X, evals

    def verbose_display(self):
        return ''

    def log_header(self):
        return []

    def log(self):
        return []