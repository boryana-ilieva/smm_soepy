import pandas as pd
import numpy as np

# Turn off pandas warning in line 22
pd.options.mode.chained_assignment = None


def get_moments(data):
    # Drop rows with nan due to initial condition
    data = data[data["Choice"].isna() == False]

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

    data["Wage Observed"] = data.apply(lambda row: get_observed_wage(row), axis=1)

    # Initialize moments dictionary
    moments = dict()

    # Store moments in groups as nested dictionary
    for group in [
        "Wage Distribution",
        "Wage by Educ",
        "Choice Probability",
        "Choice Probability by Educ",
    ]:
        moments[group] = dict()

    # Compute unconditional moments of the wage distribution
    info = data.groupby(["Period"])["Wage Observed"].describe().to_dict()

    # Save mean and standard deviation of wages for each period
    # to Wage Distribution section of the moments dictionary
    for period in sorted(data["Period"].unique().tolist()):
        if pd.isnull(info["std"][period]):
            continue
        moments["Wage Distribution"][period] = []
        for label in ["mean", "std"]:
            moments["Wage Distribution"][period].append(info[label][period])

    # Compute moments of the wage distribution by education
    info = (
        data.groupby(["Period", "Years of Education"])["Wage Observed"]
        .describe()
        .to_dict()
    )

    for period in sorted(data["Period"].unique().tolist()):
        for educ_level in sorted(data["Years of Education"].unique().tolist()):
            moments["Wage by Educ"][(period, educ_level)] = []
            try:
                if pd.isnull(info["std"][(period, educ_level)]):
                    continue
                for label in ["mean", "std"]:
                    moments["Wage by Educ"][(period, educ_level)].append(
                        info[label][(period, educ_level)]
                    )
            # Moments set to zero for periods and educ_level
            # where moments cannot be calculated
            except KeyError:
                for i in range(2):
                    moments["Wage by Educ"][(period, educ_level)].append(0.0)

    # Compute unconditional moments of the choice probabilities
    info = data.groupby(["Period"])["Choice"].value_counts(normalize=True).to_dict()

    for period in sorted(data["Period"].unique().tolist()):
        moments["Choice Probability"][period] = []
        for choice in sorted(data["Choice"].unique().tolist()):
            try:
                stat = info[(period, choice)]
            except KeyError:
                stat = 0.00
            moments["Choice Probability"][period].append(stat)

    # Compute moments of choice probabilities by education
    info = (
        data.groupby(["Period", "Years of Education"])["Choice"]
        .value_counts(normalize=True)
        .to_dict()
    )

    # Compute moments of the choice probabilities by education
    for period in sorted(data["Period"].unique().tolist()):
        for educ_level in sorted(data["Years of Education"].unique().tolist()):
            moments["Choice Probability by Educ"][(period, educ_level)] = []
            for choice in sorted(data["Choice"].unique().tolist()):
                try:
                    stat = info[(period, educ_level, choice)]
                except KeyError:
                    stat = 0.00
                moments["Choice Probability by Educ"][(period, educ_level)].append(stat)

    return moments
