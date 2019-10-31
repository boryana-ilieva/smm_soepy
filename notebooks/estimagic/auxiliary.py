from functools import partial

import pandas as pd
import numpy as np

import soepy
from estimagic.optimization.optimize import minimize

def pre_process_data(data):
    """This function perform auxiliary calculations given a data set
    and generates additional variable needed for the calculation of
    moments"""

    # Determine the education level given years of experience
    data["Educ_Level"] = 0
    data.loc[data["Years_of_Education"] == 11, "Educ_Level"] = 1
    data.loc[data["Years_of_Education"] == 12, "Educ_Level"] = 2

    # Determine the observed wage given period choice
    data["Wage_Observed"] = 0
    data.loc[data["Choice"] == 0, "Wage_Observed"] = data.loc[data["Choice"] == 0, "Period_Wage_N"]
    data.loc[data["Choice"] == 1, "Wage_Observed"] = data.loc[data["Choice"] == 1, "Period_Wage_P"]
    data.loc[data["Choice"] == 2, "Wage_Observed"] = data.loc[data["Choice"] == 2, "Period_Wage_F"]

    return data


def get_moments(data):
    # Initialize moments dictionary
    moments = dict()

    # Store moments in groups as nested dictionary
    for group in [
        "Wage_Distribution",
        "Choice_Probability",
    ]:
        moments[group] = dict()

    # Compute unconditional moments of the wage distribution
    info = data.groupby(["Period"])["Wage_Observed"].describe().to_dict()

    # Save mean and standard deviation of wages for each period
    # to Wage Distribution section of the moments dictionary
    for period in range(40):  ## TO DO: Remove hard coded number
        moments["Wage_Distribution"][period] = []
        try:
            for label in ["mean", "std"]:
                moments["Wage_Distribution"][period].append(info[label][period])
        except KeyError:
            for i in range(2):
                moments["Wage_Distribution"][period].append(
                    0.0
                )

    # Compute unconditional moments of the choice probabilities
    info = data.groupby(["Period"])["Choice"].value_counts(normalize=True).to_dict()

    for period in range(40):  ## TO DO: Remove hard coded number
        moments["Choice_Probability"][period] = []
        for choice in range(3):
            try:
                stat = info[(period, choice)]
            except KeyError:
                stat = 0.00
            moments["Choice_Probability"][period].append(stat)

    return moments

def get_weighting_matrix(data, num_agents_smm, num_samples):
    """Calculates the weighting matrix based on the
    moments of the observed data"""

    moments_sample = []

    # Collect n samples of moments
    for k in range(num_samples):
        data_sample = data.sample(n=num_agents_smm)

        moments_sample_k = get_moments(data_sample)

        moments_sample.append(moments_sample_k)

        k = +1

    # Append samples to a list of size num_samples
    # containing number of moments values each
    stats = []

    for moments_sample_k in moments_sample:
        stats.append(moments_dict_to_list(moments_sample_k))

    # Calculate sample variances for each moment
    moments_var = np.array(stats).var(axis=0)

    # Handling of nan
    moments_var[np.isnan(moments_var)] = np.nanmax(moments_var)

    # Handling of zero variances
    is_zero = moments_var <= 1e-10
    moments_var[is_zero] = 0.1

    # Construct weighting matrix
    weighting_matrix = np.diag(moments_var ** (-1))

    return weighting_matrix


def moments_dict_to_list(moments_dict):
    """This function constructs a list of available moments based on the moment dictionary."""
    moments_list = []
    for group in [
        "Wage_Distribution",
        "Choice_Probability",
    ]:
        for period in sorted(moments_dict[group].keys()):
            moments_list.extend(moments_dict[group][period])
    return moments_list


def get_objective_estimagic(model_spec_init_file_name, moments_obs, weighting_matrix, data_df, num_evals, max_evals,
                            log_file_name, params):
    # Generate simulated data set
    data_frame_sim = soepy.simulate(params, model_spec_init_file_name)

    # Process data frame
    data_frame_sim = pre_process_data(data_frame_sim)

    # Calculate simulated moments
    moments_sim = get_moments(data_frame_sim)

    # Obtain criterion function value
    fval, stats_obs, stats_sim = calculate_criterion_func_value_estimagic(moments_sim, moments_obs, weighting_matrix,
                                                                          num_evals)

    print(fval)

    # Save params and function value as pickle object.
    is_start = data_df is None

    if is_start:
        data = {'current': fval, 'start': fval, 'step': fval}
        data_df = pd.DataFrame(data, columns=['current', 'start', 'step'], index=[0])
        params.to_pickle('step.soepy.pkl')
    else:
        is_step = data_df['step'].iloc[-1] > fval
        step = data_df['step'].iloc[-1]
        start = data_df['start'].loc[0]

        if is_step:
            data = {'current': fval, 'start': start, 'step': fval}
            params.to_pickle('step.soepy.pkl')
        else:
            data = {'current': fval, 'start': start, 'step': step}

        data_df = data_df.append(data, ignore_index=True)

    _logging_smm(stats_obs, stats_sim, fval, params, weighting_matrix, num_evals, log_file_name)

    #     num_evals = num_evals + 1
    #     print(num_evals)

    #     # We want to be able to terminate in a very controlled fashion after a given set of
    #     # function evaluations.
    #     if num_evals >= max_evals:
    #         raise RuntimeError('maximum number of evaluations reached')

    return fval


def calculate_criterion_func_value_estimagic(moments_sim, moments_obs, weighting_matrix, num_evals):
    # Move all moments from a dictionary to an array
    stats_obs, stats_sim = [], []

    for group in [
        "Wage_Distribution",
        "Choice_Probability",
    ]:
        for key_ in moments_obs[group].keys():
            stats_obs.extend(moments_obs[group][key_])
            stats_sim.extend(moments_sim[group][key_])

    # Construct criterion value
    stats_dif = np.array(stats_obs) - np.array(stats_sim)

    fval = float(np.dot(np.dot(stats_dif, weighting_matrix), stats_dif))

    return fval, stats_obs, stats_sim

def _logging_smm(stats_obs, stats_sim, fval, params, weighting_matrix, num_evals, log_file_name):
    """This method contains logging capabilities that are just relevant for the SMM routine."""
    fname = log_file_name
    fname2 = "compact_" + log_file_name

    with open(fname, "a+") as outfile:
        fmt_ = "\n\n{:>8}{:>15}\n\n"
        outfile.write(fmt_.format("EVALUATION", num_evals))
        fmt_ = "\n\n{:>8}{:>15}\n\n"
        outfile.write(fmt_.format("fval", fval))
        for x in params.index:
            info = [x[0],x[1],params.loc[x,"value"]]
            fmt_ = "{:>8}" + "{:>15}" * 2 +"\n\n"
            outfile.write(fmt_.format(*info))

        fmt_ = "{:>8}" + "{:>15}" * 4 + "\n\n"
        info = ["Moment", "Observed", "Simulated", "Difference", "Weight"]
        outfile.write(fmt_.format(*info))
        for x in enumerate(stats_obs):
            stat_obs, stat_sim = stats_obs[x[0]], stats_sim[x[0]]
            info = [
                x[0],
                stat_obs,
                stat_sim,
                abs(stat_obs - stat_sim),
                weighting_matrix[x[0], x[0]],
            ]

            fmt_ = "{:>8}" + "{:15.5f}" * 4 + "\n"
            outfile.write(fmt_.format(*info))

        with open(fname2, "a+") as outfile:
            fmt_ = "\n\n{:>8}{:>15}\n\n"
            outfile.write(fmt_.format("EVALUATION", num_evals))
            fmt_ = "\n\n{:>8}{:>15}\n\n"
            outfile.write(fmt_.format("fval", fval))
            for x in params.index:
                info = [x[0], x[1], params.loc[x, "value"]]
                fmt_ = "{:>8}" + "{:>15}" * 2 + "\n\n"
                outfile.write(fmt_.format(*info))


def prepare_estimation(model_params_init_file_name, data_file_name, lower, upper):
    """Prepares objects for SMM estimation."""

    # Read in data and init file sources
    model_params_df = pd.read_pickle(model_params_init_file_name)
    data = pd.read_pickle(data_file_name)
    model_params_df["lower"] = lower
    model_params_df["upper"] = upper

    # Get moments from observed data
    data = pre_process_data(data)
    moments_obs = get_moments(data)

    # Calculate weighting matrix based on bootstrap variances of observed moments
    weighting_matrix = get_weighting_matrix(data, num_agents_smm=500, num_samples=200)

    return moments_obs, weighting_matrix, model_params_df


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


model_params_init_file_name = "params_init_delta0.pkl"
model_spec_init_file_name = "model_spec_init_test_zero.yml"
data_file_name = "data_obs_3types.pkl"

moments_obs, weighting_matrix, model_params_df = prepare_estimation(model_params_init_file_name, data_file_name, lower, upper)

# Prepare interface for minimization routine
data_df = None
num_evals = 0
max_evals = 2

log_file_name = "monitoring.estimagic.smm.test_server.info"

objective = partial(get_objective_estimagic, model_spec_init_file_name, moments_obs, weighting_matrix, data_df, num_evals, max_evals, log_file_name)

algo_options = {
    "stopeval": 1e-9,
    "maxeval": 1000,
}

# Estimate
result = minimize(criterion = objective, params = model_params_df, algorithm = "nlopt_bobyqa", algo_options = algo_options)