import logging
import yaml

import os
import sys
sys.path.append(os.path.abspath(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]))


import pybobyqa
import numpy as np
import pandas as pd
from functools import partial

import soepy
from smm.objective import get_objective
from smm.moments import get_moments
from smm.moments_extended import get_moments_extended
from smm.weighting import get_weighting_matrix
from smm.weighting_extended import get_weighting_matrix_extended

# Specify init file
init_file_name = 'toy_model_init_file_1000.yml'

# Get observed moments
data_frame_observed = pd.read_csv('toy_model_sim_benchmark.csv', sep = '\t')
moments_obs = get_moments(data_frame_observed)
# git moments_obs_list = moments_dict_to_list_extended(moments_obs)

# Get weighting matrix
weighting_matrix = get_weighting_matrix(data_frame_observed, num_agents_smm = 1000, num_samples = 100)

# Define objective function as a function of the parameter vector only
objective = partial(get_objective, init_file_name, moments_obs, weighting_matrix)

# Define starting values
optim_paras_start = np.tile((5.406,
              5.574,
              6.949,
              0.152,
              0.229,
              0.306,
              0.150,
              0.096,
              0.116,
              0.081,
              0.057,
              0.073,
              -0.10,
              -0.30,
              1.00,
              1.25,
              1.60), 1)


lower = optim_paras_start*0.9
upper = optim_paras_start*1.1

# Log
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Optimize
soln = pybobyqa.solve (objective,
       optim_paras_start,
       rhobeg=0.01,
       rhoend=1e-4,
       npt = 19,
       maxfun=2,
       bounds=(lower,upper),
       scaling_within_bounds=True
       )

print(soln)


# print("")
# print("** SciPy results **")
# print("Solution xmin = %s" % str(soln.x))
# print("Objective value f(xmin) = %.10g" % (soln.fun))
# print("Needed %g objective evaluations" % soln.nfev)
# print("Exit flag = %g" % soln.status)
# print(soln.message)
