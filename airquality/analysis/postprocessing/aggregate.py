import os
import numpy as np
import pandas as pd
from parse import parse

import scipy.stats
import statsmodels.stats.api as sms
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import zscore

score_funcs = [r2_score, mean_absolute_error]

def iqr_score(x):
    ''' For outlier detection - outliers are 1.5 x IQR above or below
        the upper or lower quartiles. 
    '''
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    q1_score = (x - q1).abs() / iqr
    q3_score = (x - q3).abs() / iqr

    return np.maximum(q1_score, q3_score)

def trial_fold_index(path):
    ''' Label trial number and fold number by parsing the path, for use
        in aggregation routines
    '''
    trial_idx, fold_idx = 0, 0
    for folder in path.split('/'):
        res = parse('fold_{:02d}', folder)
        if res: fold_idx = res[0]

        res = parse('trial_{:02d}', folder)
        if res: trial_idx = res[0]

    return trial_idx, fold_idx

def aggregate_n_params(path):
    ''' Compute the average number of parameters from a model
    '''
    n_params = []
    for root, dirs, files in os.walk(path):
        if 'model_summary.csv' in files:
            n_params.append(np.sum(pd.read_csv(os.path.join(root, 'model_summary.csv'))['coef'] != 0))
    
    return np.mean(n_params), np.std(n_params)

def aggregate_effects(path, 
                      mean_column='risk_ratio (est.)', 
                      ci_columns=['lower_ci (est.)', 'upper_ci (est.)'],
                      overwrite=False,
                      remove_outliers=None):
    ''' Aggregate the mean and variance across multiple trials
        Here, E[x] = \sum_i E[x_i] / N
        Var[x] = E[Var[x_i]] + Var[E[x_i]]
    '''
    if not overwrite and os.path.exists(f'{path}/trial_effects.csv'):
        return pd.read_csv(f'{path}/trial_effects.csv')
    
    risk_df = []
    for root, dirs, files in os.walk(path):
        if 'model_summary.csv' in files:
            path_df = pd.read_csv(os.path.join(root, 'model_summary.csv'))
            trial_idx, fold_idx = trial_fold_index(root)
            path_df['trial_index'] = trial_idx
            path_df['fold_index'] = fold_idx
            risk_df.append(path_df)

    risk_df = pd.concat(risk_df, axis=0)
    if remove_outliers == 'iqr':
        risk_df['iqr_score'] = risk_df.groupby('parameter')[mean_column].transform(lambda x : iqr_score(x))
        risk_df = risk_df[risk_df['iqr_score'].abs() <= 1.5]
    elif remove_outliers == 'zscore':
        risk_df['z_score'] = risk_df.groupby('parameter')[mean_column].transform(lambda x : zscore(x))
        risk_df = risk_df[risk_df['z_score'].abs() <= 3]
    
    # Estimate variance for each trial using reported confidence intervals
    risk_df['mean'] = risk_df[mean_column]
    risk_df['variance'] = 0.
    for CI_col in ci_columns:
        risk_df['variance'] += np.power(np.abs(risk_df['mean'] - risk_df[CI_col]) / 1.96, 2) / len(ci_columns)
    
    # Aggregate the mean and variance over all trials
    agg_df = risk_df.groupby('parameter').agg({
        'mean': 'mean',
        'variance': 'mean',
    })
    agg_df['variance'] = agg_df['variance'] \
        + risk_df.groupby('parameter')['mean'].agg(lambda x : x.pow(2).mean()) \
        - agg_df['mean'].pow(2)
    
    agg_df.to_csv(f'{path}/trial_effects.csv')
    
    return agg_df.reset_index()

def aggregate_predictions(path, 
                          overwrite=False, 
                          csv_kwargs = {'converters': {'date_local': lambda x: pd.Period(x, freq='D')}}):
    ''' Aggregate predicted counts across multiple trials
    '''
    if not overwrite and os.path.exists(f'{path}/trial_predictions.csv'):
        return pd.read_csv(f'{path}/trial_predictions.csv', **csv_kwargs)
    
    trial_dfs = []
    for root, dirs, files in os.walk(path):
        if 'model_predictions.csv' in files:
            path_df = pd.read_csv(os.path.join(root, 'model_predictions.csv'), **csv_kwargs)
            trial_idx, fold_idx = trial_fold_index(root)
            path_df['trial_index'] = trial_idx
            trial_dfs.append(path_df)

    trial_dfs = pd.concat(trial_dfs, axis=0)
    trial_dfs = trial_dfs.groupby(['date_local', 'trial_index'])[['Observed', 'Predicted']].sum()
    trial_dfs.to_csv(f'{path}/trial_predictions.csv')
    return trial_dfs