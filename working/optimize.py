import logging

import os
import sys

# Set path to root directory smm_soepy
sys.path.append(
    os.path.abspath(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
)

import pybobyqa
import numpy as np
from functools import partial

from smm.smm_auxiliary import pre_process_soep_data
from smm.objective import get_objective
from smm.moments import get_moments_obs
from smm.weighting import get_weighting_matrix


# Get observed moments

# Server path
# data_frame_observed = pre_process_soep_data('/projekte/bilieva/homes/data/soepcore_prep.dta')

# Mac path
data_frame_observed = pre_process_soep_data('/Users/boryanailieva/Projects/Data/soepcore_prep.dta')

moments_obs = get_moments_obs(data_frame_observed)

# Get weighting matrix
weighting_matrix = get_weighting_matrix(
    data_frame_observed, num_agents_smm=6000, num_samples=200
)

# Specify init file
init_file_name = "sim_toy_model_init_file_1000.yml"

# # Define objective function as a function of the parameter vector only
objective = partial(get_objective, init_file_name, moments_obs, weighting_matrix)

# ##############################
# # Test
#
# # Define starting values
# optim_paras = np.tile(
#     (
#         1.792,
#         1.808,
#         1.856,
#         0.112,
#         0.199,
#         0.266,
#         0.150,
#         0.096,
#         0.116,
#         0.081,
#         0.057,
#         0.073,
#         1.88,
#         2.33,
#         -0.200,
#         -0.500,
#         0.5,
#         0.010,
#         0.200,
#         0.400,
#     ),
#     1,
# )
#
# crit_function_value = get_objective(
#     init_file_name, moments_obs, weighting_matrix, optim_paras
# )
#
# print(crit_function_value)
#
# ##############################


# Define starting values
optim_paras_start = np.tile(
    (
        1.792,
        1.808,
        1.856,
        0.112,
        0.199,
        0.266,
        0.20,
        0.20,
        0.20,
        0.081,
        0.057,
        0.073,
        1.88,
        2.33,
        -0.200,
        -0.500,
        0.5,
        0.010,
        0.200,
        0.400,
    ),
    1,
)


# lower = optim_paras_start * 0.4
# upper = optim_paras_start * 1.7

lower = np.tile(
    (
        1.000,
        1.000,
        1.000,
        0.050,
        0.050,
        0.050,
        0.005,
        0.005,
        0.005,
        0.001,
        0.001,
        0.001,
        1.00,
        1.00,
        -0.400,
        -0.800,
        0.001,
        0.001,
        0.001,
        0.001,
    ),
    1,
)

upper = np.tile(
    (
        3.000,
        3.000,
        3.000,
        0.400,
        0.400,
        0.400,
        0.600,
        0.600,
        0.600,
        0.150,
        0.150,
        0.150,
        4.00,
        4.00,
        -0.050,
        -0.150,
        0.999,
        0.800,
        0.800,
        0.800,
    ),
    1,
)


# Log
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Optimize
soln = pybobyqa.solve(
    objective,
    optim_paras_start,
    rhobeg=0.01,
    rhoend=1e-9,
    maxfun=5000,
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
