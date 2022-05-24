#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cma.objective_function.mixed as f_mixed
import cma.optimizer.cmaeswm as cma
import cma.util.sampler as sampler
import cma.util.weight as weight
import cma.util.log as logg

def cma_run(): 
    # set function
    # function list: 'SphereOneMax','SphereLeadingOnes','EllipsoidOneMax','EllipsoidLeadingOnes','SphereInt','EllipsoidInt'
    func = 'SphereOneMax'
    dim = 40     # total number of dimensions
    dim_bi = dim//2     # number of binary variables
    dim_in = dim//2     # number of integer variables
    max_evals = dim * 1e4
    f = eval('f_mixed.' + func)(d=dim, bid=dim_bi, ind=0, max_eval=max_evals)     # continuous + binary function
    # f = eval('f_mixed.' + func)(d=dim, bid=0, ind=dim_in, max_eval=max_evals)     # continuous + integer function

    # discrete space
    discrete_space = np.tile(np.arange(0, 2, 1), (dim_bi, 1))     # binary variables
    # discrete_space = np.tile(np.arange(-10, 11, 1), (dim_in, 1))     # integer variables

    # hyperparameter (lambda, weight function, margin)
    lam = cma.CMAParam.pop_size(dim)
    w_func = weight.CMAWeightWithNegativeWeights(lam, dim, min_problem=f.minimization_problem)
    margin = 1 / (dim * lam)

    # output log file (csv)
    result_folder = f'./output/'
    path_name = result_folder + f'{func}/dim{dim}/'
    output = logg.DataLogger(file_name='result.csv', path_name=path_name)

    # initial values of Gaussian distribution
    init_m, init_sigma = f_mixed.initial_setting_for_gaussian(f)

    # optimizer
    samp = sampler.Sampler(f, lam)
    opt = cma.CMAESwM(dim, discrete_space, w_func, samp,
                           lam=lam, m=init_m, sigma=init_sigma,
                           margin=margin, restart=-1, minimal_eigenval=1e-30)

    # run
    result = opt.run(samp,
         logger=output,
         verbose=True)

    print(f'is_sucess: {result[2]}')


if __name__ == '__main__':
    cma_run()