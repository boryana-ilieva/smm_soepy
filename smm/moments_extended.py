import pandas as pd
import numpy as np
import pickle as pkl
import json


def get_moments_extended(data_frame):

    # Determine the observed wage given period choice
    def get_observed_wage(row):
        if row["Choice"] == 2:
            return row["Period Wage F"]
        elif row["Choice"] == 1:
            return row["Period Wage P"]
        elif row["Choice"] == 0:
            return row["Period Wage N"]
        else:
            return np.nan

    data_frame["Wage Observed"] = data_frame.apply(
        lambda row: get_observed_wage(row), axis=1
    )

    # Determine years of part-time experience
    # by person and period
    def get_part_time_ind(row):
        if row["Choice"] == 1:
            return 1
        elif row["Choice"] == 2 or row["Choice"] == 0:
            return 0
        else:
            return np.nan

    data_frame["Part Time Ind"] = data_frame.apply(
        lambda row: get_part_time_ind(row), axis=1
    )

    # Determine years of full-time experience
    # by person and period
    def get_full_time_ind(row):
        if row["Choice"] == 2:
            return 1
        elif row["Choice"] == 1 or row["Choice"] == 0:
            return 0
        else:
            return np.nan

    data_frame["Full Time Ind"] = data_frame.apply(
        lambda row: get_full_time_ind(row), axis=1
    )

    # Extract the level of education
    # given years of education
    def get_educ_level(row):
        if row["Years of Education"] <= 10:
            return 0

        elif row["Years of Education"] > 10 and row["Years of Education"] <= 12:
            return 1

        else:
            return 2

    data_frame["Educ Level"] = data_frame.apply(lambda row: get_educ_level(row), axis=1)

    # Create cumulated years of part-time and full-time experience
    data_frame["PT CS"] = data_frame.groupby(["Identifier"]).cumsum()["Part Time Ind"]
    data_frame["FT CS"] = data_frame.groupby(["Identifier"]).cumsum()["Full Time Ind"]

    # Initialize moments dictionary
    moments = dict()

    # Store moments in groups as nested dictionary
    for group in [
        "Wage Distribution",
        "Wage by Educ",
        "Wage by FT Exp",
        "Wage by PT Exp",
        "Choice Probability",
        "Choice Probability by Educ",
        "Choice Probability by FT Exp",
        "Choice Probability by PT Exp",
    ]:
        moments[group] = dict()

    # Compute wage distribution moments
    info = data_frame.groupby(["Period"])["Wage Observed"].describe().to_dict()

    # Save mean and standard diviation of wages for each period
    # to Wage Distribution section of the moments dictionary
    for period in sorted(data_frame["Period"].unique().tolist()):
        if pd.isnull(info["std"][period]):
            continue
        moments["Wage Distribution"][period] = []
        for label in ["mean", "std"]:
            moments["Wage Distribution"][period].append(info[label][period])

    # Compute information about the choice probabilities
    info = (
        data_frame.groupby(["Period"])["Choice"].value_counts(normalize=True).to_dict()
    )

    for period in sorted(data_frame["Period"].unique().tolist()):
        moments["Choice Probability"][period] = []
        for choice in range(1, 3):
            try:
                stat = info[(period, choice)]
            except KeyError:
                stat = 0.00
            moments["Choice Probability"][period].append(stat)

    # Compute moments on wage distribution by education
    info_test1 = (
        data_frame.groupby(["Period", "Educ Level"])["Wage Observed"]
        .describe()
        .to_dict()
    )

    for period in sorted(data_frame["Period"].unique().tolist()):
        for educ_level in sorted(data_frame["Educ Level"].unique().tolist()):
            if pd.isnull(info_test1["std"][(period, educ_level)]):
                continue
            moments["Wage by Educ"][(period, educ_level)] = []
            for label in ["mean", "std"]:
                moments["Wage by Educ"][(period, educ_level)].append(
                    info_test1[label][(period, educ_level)]
                )

    # Compute moments on wage means by years of part-time experience
    info_test2 = data_frame.groupby(["PT CS"])["Wage Observed"].mean().to_dict()

    for part_time_exp in sorted(data_frame["Period"].unique().tolist()):
        try:
            stat = info_test2[part_time_exp]
        except KeyError:
            stat = 0.00
        moments["Wage by PT Exp"][part_time_exp] = []
        moments["Wage by PT Exp"][part_time_exp].append(stat)

    # Compute moments on wage means by years of full-time experience
    info_test3 = data_frame.groupby(["FT CS"])["Wage Observed"].mean().to_dict()

    for full_time_exp in sorted(data_frame["Period"].unique().tolist()):
        try:
            stat = info_test3[full_time_exp]
        except KeyError:
            stat = 0.00
        moments["Wage by FT Exp"][full_time_exp] = []
        moments["Wage by FT Exp"][full_time_exp].append(stat)

    # Compute moments on choice probabilities by level of education
    info_test4 = (
        data_frame.groupby(["Period", "Educ Level"])["Choice"]
        .value_counts(normalize=True)
        .to_dict()
    )

    for period in sorted(data_frame["Period"].unique().tolist()):
        for educ_level in sorted(data_frame["Educ Level"].unique().tolist()):
            moments["Choice Probability by Educ"][(period, educ_level)] = []
            for choice in range(1, 3):
                try:
                    stat = info_test4[(period, educ_level, choice)]
                except KeyError:
                    stat = 0.00
                moments["Choice Probability by Educ"][(period, educ_level)].append(stat)

    # Compute moments on choice probabilities by years of part-time experience
    info_test5 = (
        data_frame.groupby(["PT CS"])["Choice"].value_counts(normalize=True).to_dict()
    )

    for part_time_exp in sorted(data_frame["Period"].unique().tolist()):
        moments["Choice Probability by PT Exp"][part_time_exp] = []
        for choice in range(1, 3):
            try:
                stat = info_test5[(part_time_exp, choice)]
            except KeyError:
                stat = 0.00
            moments["Choice Probability by PT Exp"][(part_time_exp)].append(stat)

    # Compute moments on choice probabilities by years of full-time experience
    info_test6 = (
        data_frame.groupby(["FT CS"])["Choice"].value_counts(normalize=True).to_dict()
    )

    for full_time_exp in sorted(data_frame["Period"].unique().tolist()):
        moments["Choice Probability by FT Exp"][full_time_exp] = []
        for choice in range(1, 3):
            try:
                stat = info_test6[(full_time_exp, choice)]
            except KeyError:
                stat = 0.00
            moments["Choice Probability by FT Exp"][(full_time_exp)].append(stat)

    # Save moments dictionary to file
    # fname = 'moments.soepy.extended.'
    # json.dump(moments, open(fname + 'json', 'w'), indent=4, sort_keys=True)
    # pkl.dump(moments, open(fname + 'pkl', 'wb'))

    return moments
