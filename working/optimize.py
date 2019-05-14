import logging

import os
import sys

# Set path to root directory smm_soepy
sys.path.append(
    os.path.abspath(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
)

import pybobyqa
import numpy as np
import pandas as pd
from functools import partial

from smm.objective import get_objective
from smm.moments import get_moments
from smm.weighting import get_weighting_matrix


# Get observed moments
data_frame_observed = pd.read_pickle("toy_model_sim_benchmark.pkl")
moments_obs = get_moments(data_frame_observed)

# Get weighting matrix
weighting_matrix = get_weighting_matrix(
    data_frame_observed, num_agents_smm=1000, num_samples=100
)

# Specify init file
init_file_name = "toy_model_init_file_1000.yml"

# Define objective function as a function of the parameter vector only
objective = partial(get_objective, init_file_name, moments_obs, weighting_matrix)

###############################
# # Test
#
# # Define starting values
# optim_paras = np.tile(
#     (
#         5.406,
#         5.574,
#         6.949,
#         0.152,
#         0.229,
#         0.306,
#         0.150,
#         0.096,
#         0.116,
#         0.081,
#         0.057,
#         0.073,
#         -0.10,
#         -0.30,
#         1.00,
#         1.25,
#         1.60,
#     ),
#     1,
# )
#
# crit_function_value = get_objective(
#     init_file_name, moments_obs, weighting_matrix, optim_paras
# )
#
# print(crit_function_value)

###############################


# Define starting values
optim_paras_start = np.tile(
    (
        5.406,
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
        1.60,
    ),
    1,
)


lower = optim_paras_start * 0.9
upper = optim_paras_start * 1.1

# Log
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Optimize
soln = pybobyqa.solve(
    objective,
    optim_paras_start,
    rhobeg=0.01,
    rhoend=1e-4,
    maxfun=2,
    bounds=(lower, upper),
    scaling_within_bounds=True,
)

print(soln)


# print("")
# print("** SciPy results **")
# print("Solution xmin = %s" % str(soln.x))
# print("Objective value f(xmin) = %.10g" % (soln.fun))
# print("Needed %g objective evaluations" % soln.nfev)
# print("Exit flag = %g" % soln.status)
# print(soln.message)
