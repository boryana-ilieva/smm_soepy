import pandas as pd
import numpy as np
import pickle as pkl
import json

def get_moments (data_frame):

    #Determine the observed wage given period choice
    def get_observed_wage (row):
        if row['Choice'] == 2:
            return row['Period Wage F']
        elif row['Choice'] ==1:
            return row['Period Wage P']
        elif row['Choice'] ==0:
            return row['Period Wage N']
        else:
            return np.nan
        
    data_frame['Wage Observed'] = data_frame.apply (lambda row: get_observed_wage (row),axis=1)

    #Initialize moments dictionary
    moments = dict()

    #Store moments in groups as nested dictionary
    for group in ['Wage Distribution', 'Choice Probability']:
        moments[group] = dict()

    # Compute wage distribution moments
    info = data_frame.groupby(['Period'])['Wage Observed'].describe().to_dict()

    # Save mean and standard diviation of wages for each period
    # to Wage Distribution section of the moments dictionary
    for period in sorted(data_frame['Period'].unique().tolist()):
        if pd.isnull(info['std'][period]):
            continue
        moments['Wage Distribution'][period] = []
        for label in ['mean', 'std']:
            moments['Wage Distribution'][period].append(info[label][period])

    # Compute information about the choice probabilities
    info = data_frame.groupby(['Period'])['Choice'].value_counts(normalize=True).to_dict()

    for period in sorted(data_frame['Period'].unique().tolist()):
        moments['Choice Probability'][period] = []
        for choice in range(1, 3):
            try:
                stat = info[(period, choice)]
            except KeyError:
                stat = 0.00
            moments['Choice Probability'][period].append(stat)

    # Save moments dictionary to file
    fname = 'moments.soepy.'
    json.dump(moments, open(fname + 'json', 'w'), indent=4, sort_keys=True)
    pkl.dump(moments, open(fname + 'pkl', 'wb'))

    return moments