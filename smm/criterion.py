import numpy as np
import pickle as pkl

def calculate_criterion_func_value(moments_sim, moments_obs, weighting_matrix):

    # Supply observed moments
    #moments_obs = pkl.load(open('moments.soepy_obs.pkl', 'rb'))
    
    # Move all moments from a dictionary to an array
    stats_obs, stats_sim = [], []
    
    for group in ['Wage Distribution', 'Choice Probability']:
        for key_ in moments_obs[group].keys():
            stats_obs.extend(moments_obs[group][key_])
            stats_sim.extend(moments_sim[group][key_])
    
    # Construct criterion value
    stats_dif = np.array(stats_obs) - np.array(stats_sim)
    
    func = float(np.dot(np.dot(stats_dif, weighting_matrix), stats_dif))
    
    return func