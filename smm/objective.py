import soepy
from smm.moments import get_moments
from smm.moments_extended import get_moments_extended
from smm.criterion import calculate_criterion_func_value
from smm.smm_auxiliary import update_optim_paras


def get_objective(init_file_name, moments_obs, weighting_matrix, optim_paras):

    print(optim_paras)

    # Update parameter
    update_optim_paras(init_file_name, optim_paras)

    # Generate simulated data set
    data_frame = soepy.simulate("smm_init_file.soepy.yml")

    # Calculate simulated moments
    moments_sim = get_moments(data_frame)

    # Obtain criterion function value
    func = calculate_criterion_func_value(moments_sim, moments_obs, weighting_matrix)

    return func
