import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from sklearn.metrics import r2_score, mean_absolute_error
from ..postprocessing.aggregate import iqr_score

def plot_aggregate_predictions(ax, 
                               key : str, 
                               n_params : float, 
                               trial_predictions : pd.DataFrame, 
                               color : str, 
                               marker : str, 
                               std_params : float = None, 
                               plot_observed : bool = False, 
                               remove_outliers : bool = True):
    agg_df = trial_predictions.groupby('date_local')[['Observed', 'Predicted']].mean()
    score_funcs = [r2_score, mean_absolute_error]

    # Plot the predictions
    if plot_observed:
        ax[0,0].plot(agg_df.index.to_timestamp(), agg_df['Observed'], color='black', label='Observed')
    ax[0,0].plot(agg_df.index.to_timestamp(), agg_df['Predicted'], color=color, label=key)

    # Plot scores as a function of parameters
    for j, func in enumerate(score_funcs):
        scores = trial_predictions.groupby('trial_index').apply(lambda group : func(group['Observed'], group['Predicted']), include_groups=False)
        scores = pd.DataFrame({'score': scores.values}, index=scores.index)
        if remove_outliers and len(scores) > 2:
            scores['iqr_score'] = iqr_score(scores['score'])
            scores = scores[scores['iqr_score'].abs() <= 1.5]
        ax[j, 1].errorbar(
            n_params, 
            scores['score'].mean(), 
            xerr=std_params,
            yerr=scores['score'].std(), 
            capsize=2, 
            label=key, 
            marker=marker, 
            color=color
        )