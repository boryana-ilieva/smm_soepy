import numpy as np

from estimagic.optimization.optimize import minimize

from notebooks.estimagic.auxiliary import prepare_estimation
from notebooks.estimagic.auxiliary import get_moments
from notebooks.estimagic.SimulationBasedEstimation import SimulationBasedEstimationCls


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
        -0.400,
        -0.800,
        0.001,
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
        -0.050,
        -0.150,
        0.999,
        0.999,
        0.800,
        0.800,
        0.800,
    ),
    1,
)

model_params_init_file_name = "init_files/final_delta0_patams.pkl"
model_spec_init_file_name = "init_files/model_spec_init_test_zero.yml"
data_file_name = "init_files/data_obs_3types.pkl"
log_file_name_extension = "test_delta0_start"

moments_obs, weighting_matrix, model_params_df = prepare_estimation(
    model_params_init_file_name, model_spec_init_file_name, data_file_name, lower, upper
)

max_evals = 2000

adapter_smm = SimulationBasedEstimationCls(
    params=model_params_df,
    model_spec_init_file_name=model_spec_init_file_name,
    moments_obs=moments_obs,
    weighting_matrix=weighting_matrix,
    get_moments=get_moments,
    log_file_name_extension=log_file_name_extension,
    max_evals=max_evals,
)

algo_options = {"stopeval": 1e-9}


result = minimize(
    criterion=adapter_smm.get_objective,
    params=adapter_smm.params,
    algorithm="nlopt_bobyqa",
    algo_options=algo_options,
)
