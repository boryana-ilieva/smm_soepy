import collections
import yaml

import pandas as pd
import numpy as np


def update_optim_paras(init_file_name, optim_paras):
    """Update the parameters with values of 
    the current optimizer iteration"""

    with open(init_file_name) as y:
        init_dict = yaml.load(y, Loader=yaml.FullLoader)

    init_dict["PARAMETERS"]["gamma_0s1"] = optim_paras.item(0)
    init_dict["PARAMETERS"]["gamma_0s2"] = optim_paras.item(1)
    init_dict["PARAMETERS"]["gamma_0s3"] = optim_paras.item(2)
    init_dict["PARAMETERS"]["gamma_1s1"] = optim_paras.item(3)
    init_dict["PARAMETERS"]["gamma_1s2"] = optim_paras.item(4)
    init_dict["PARAMETERS"]["gamma_1s3"] = optim_paras.item(5)
    init_dict["PARAMETERS"]["g_s1"] = optim_paras.item(6)
    init_dict["PARAMETERS"]["g_s2"] = optim_paras.item(7)
    init_dict["PARAMETERS"]["g_s3"] = optim_paras.item(8)
    init_dict["PARAMETERS"]["delta_s1"] = optim_paras.item(9)
    init_dict["PARAMETERS"]["delta_s2"] = optim_paras.item(10)
    init_dict["PARAMETERS"]["delta_s3"] = optim_paras.item(11)
    init_dict["PARAMETERS"]["theta_p"] = optim_paras.item(12)
    init_dict["PARAMETERS"]["theta_f"] = optim_paras.item(13)
    init_dict["PARAMETERS"]["sigma_1"] = optim_paras.item(14)
    init_dict["PARAMETERS"]["sigma_2"] = optim_paras.item(15)
    init_dict["PARAMETERS"]["sigma_3"] = optim_paras.item(16)

    print_dict(init_dict)


def print_dict(init_dict, file_name="smm_init_file"):
    """This function prints the initialization dict to a *.yml file."""
    ordered_dict = collections.OrderedDict()
    order = [
        "GENERAL",
        "CONSTANTS",
        "INITIAL_CONDITIONS",
        "SIMULATION",
        "SOLUTION",
        "PARAMETERS",
    ]
    for key_ in order:
        ordered_dict[key_] = init_dict[key_]

    with open("{}.soepy.yml".format(file_name), "w") as outfile:
        yaml.dump(ordered_dict, outfile, explicit_start=True, indent=4)


def pre_process_soep_data(file_name):
    data_full = pd.read_stata(file_name)

    # Restrict sample to age 50
    data_30periods = data_full[data_full["age"] < 47]

    # Restirct sample to west Germany
    data = data_30periods[data_30periods["east"] == 0]

    # Drop observations with missing values in hdegree
    data = data[data["hdegree"].isna() == False]

    # Generate period variable
    def get_period(row):
        return row["age"] - 17

    data["Period"] = data.apply(
        lambda row: get_period(row), axis=1
    )

    # Determine the observed wage given period choice
    def recode_educ_level(row):
        if row["hdegree"] == 'Primary/basic vocational':
            return 0
        elif row["hdegree"] == 'Abi/intermediate voc.':
            return 1
        elif row["hdegree"] == 'University':
            return 2
        else:
            return np.nan

    data["Educ Level"] = data.apply(
        lambda row: recode_educ_level(row), axis=1
    )

    # Recode choice
    # Determine the observed wage given period choice
    def recode_choice(row):
        if row["empchoice"] == 'Full-Time':
            return 2
        elif row["empchoice"] == 'Part-Time':
            return 1
        elif row["empchoice"] == 'Non-Working':
            return 0
        else:
            return np.nan

    data["Choice"] = data.apply(
        lambda row: recode_choice(row), axis=1
    )

    # Generate wage for Non-Employment choice
    data["wage_nw_imp"] = 6.00

    # Determine the observed wage given period choice
    def get_observed_wage(row):
        if row["empchoice"] == 'Full-Time':
            return row["wage_ft"]
        elif row["empchoice"] == 'Part-Time':
            return row["wage_pt"]
        elif row["empchoice"] == 'Non-Working':
            return row["wage_nw_imp"]
        else:
            return np.nan

    data["Wage Observed"] = data.apply(
        lambda row: get_observed_wage(row), axis=1
    )

    return data
