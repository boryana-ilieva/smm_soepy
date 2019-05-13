import numpy as np


def calculate_criterion_func_value(moments_sim, moments_obs, weighting_matrix):

    # Move all moments from a dictionary to an array
    stats_obs, stats_sim = [], []

    for group in [
        "Wage Distribution",
        "Wage by Educ",
        "Choice Probability",
        "Choice Probability by Educ",
    ]:
        for key_ in moments_obs[group].keys():
            stats_obs.extend(moments_obs[group][key_])
            stats_sim.extend(moments_sim[group][key_])

    # Construct criterion value
    stats_dif = np.array(stats_obs) - np.array(stats_sim)

    func = float(np.dot(np.dot(stats_dif, weighting_matrix), stats_dif))

    return func
