#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod

# public symbols
__all__ = ['BaseOptimizer']


class BaseOptimizer(object):
    @abstractmethod
    def sampling_model(self):
        pass

    @abstractmethod
    def update(self, X, evals):
        """
        Abstract method for parameter updating.
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
        Return log header list.

        :return: header info list for log
        :rtype string list:
        """
        return []

    def log(self):
        """
        Return log string list.

        :return: log string list
        :rtype string list:
        """
        return []

    def run(self, sampler, logger=None, verbose=True):
        f = sampler.f
        if logger is not None:
            logger.write_csv(['Ite'] + f.info_header() + self.log_header() + sampler.log_header())

        ite = 0

        while not sampler.f.terminate_condition() and not self.terminate_condition():
            ite += 1

            # sampling and evaluation
            X, evals = sampler(self.sampling_model())

            # display and save log
            if verbose:
                print(str(ite) + f.verbose_display() + self.verbose_display() + sampler.verbose_display())
            if logger is not None:
                logger.write_csv([str(ite)] + f.info_list() + self.log() + sampler.log())

            # parameter update
            self.update(X, evals)

        return [f.eval_count, f.best_eval, f.is_success()]
