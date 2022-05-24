#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np

# public symbols
__all__ = ['ObjectiveFunction']


class ObjectiveFunction(object):
    """
    Abstract Class for objective function

    :var int eval_count: evaluation count
    :var float best_eval: best evaluation value
    :var float target_eval: target evaluation value
    :var int max_eval: maximum number of evaluations
    :var bool minimization_problem: minimization problem or not
    """
    __metaclass__ = ABCMeta

    minimization_problem = True

    def __init__(self, target_eval, max_eval):
        self.target_eval = target_eval
        self.max_eval = max_eval

        self.eval_count = 0
        self.best_eval = np.inf if self.minimization_problem else -np.inf

        self.is_better = (lambda x, y: x < y) if self.minimization_problem else (lambda x, y: x > y)
        self.is_better_eq = (lambda x, y: x <= y) if self.minimization_problem else (lambda x, y: x >= y)
        self.get_better = (lambda x, y: np.minimum(x, y)) if self.minimization_problem else (lambda x, y: np.maximum(x, y))
        self.get_best = (lambda evals: np.min(evals)) if self.minimization_problem else (lambda evals: np.max(evals))

    @abstractmethod
    def __call__(self, X):
        """
        Abstract method for evaluation.

        :param X: candidate solutions
        :return: evaluation values
        """
        pass

    def clear(self):
        self.eval_count = 0
        self.best_eval = np.inf if self.minimization_problem else -np.inf

    def terminate_condition(self):
        """
        Check terminate condition.

        :return bool: terminate condition is satisfied or not
        """
        if self.eval_count >= self.max_eval:
            return True
        return self.is_success()

    def is_success(self):
        """
        Check success (i.e, found better solution than target evaluation value).

        :return bool: success or not
        """
        if self.is_better_eq(self.best_eval, self.target_eval):
            return True
        else:
            return False

    def verbose_display(self):
        """
        Return verbose display string.

        :return str: string for verbose display
        """
        return ' EvalCount: %d' % self.eval_count + ' BestEval: {}'.format(self.best_eval)

    @staticmethod
    def info_header():
        """
        Return string list of header.

        :return: string list of header
        """
        return ['EvalCount', 'BestEval']

    def info_list(self):
        """
        Return string list of evaluation count and best evaluation value.

        :return: string list of evaluation count and best evaluation value
        """
        return ['%d' % self.eval_count, '%e' % self.best_eval]

    def _update_best_eval(self, evals):
        """
        Update best evaluation value.

        :param evals: new evaluation values
        :type evals: array_like, shape(lam), dtype=float
        """
        self.best_eval = self.get_better(self.get_best(evals), self.best_eval)
