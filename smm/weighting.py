import pandas as pd
import numpy as np

from smm.moments import get_moments


def get_weighting_matrix(data_frame, num_agents_smm, num_samples):
    """Calculates the weighting matrix based on the
    moments of the observed data"""

    moments_sample = []

    # Collect n samples of moments
    for k in range(num_samples):

        data_frame_sample = data_frame.sample(n=num_agents_smm)

        insample_periods = data_frame_sample["Period"].unique()

        moments_sample_k = get_moments(data_frame_sample)

        moments_sample.append(moments_sample_k)

        k = +1

    stats = []

    for moments_sample_k in moments_sample:
        stats.append(moments_dict_to_list(moments_sample_k))

    # Calculate sample variances for each moment
    moments_var = np.array(stats).T.var(axis=1)

    # Handling of zero variances
    is_zero = moments_var <= 1e-10
    moments_var[is_zero] = 0.1

    # Construct weighting matrix
    weighting_matrix = np.diag(moments_var ** (-1))

    return weighting_matrix


def moments_dict_to_list(moments_dict):
    """This function constructs a list of available moments based on the moment dictionary."""
    moments_list = []
    for group in ["Wage Distribution", "Choice Probability"]:
        for period in sorted(moments_dict[group].keys()):
            moments_list.extend(moments_dict[group][period])
    return moments_list
