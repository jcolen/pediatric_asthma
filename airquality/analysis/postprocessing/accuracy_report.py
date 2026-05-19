import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_absolute_error, mean_poisson_deviance

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def accuracy_report(pred_df):
    df = pred_df.groupby('date_local')[['Observed', 'Predicted']].sum()
    c = df.index.to_timestamp().year

    # Compute accuracy scores
    acc_df = pd.DataFrame(columns=['R2', 'MAE', 'Poisson Dev.'])
    acc_df.index.name = 'Year'

    for year in sorted(c.unique()):
        args = df.loc[c == year, 'Observed'], df.loc[c == year, 'Predicted']
        acc_df.loc[year, 'R2'] = r2_score(*args)
        acc_df.loc[year, 'MAE'] = mean_absolute_error(*args)
        acc_df.loc[year, 'Poisson Dev.'] = mean_poisson_deviance(*args)

    args = df['Observed'], df['Predicted']
    acc_df.loc['Overall', 'R2'] = r2_score(*args)
    acc_df.loc['Overall', 'MAE'] = mean_absolute_error(*args)
    acc_df.loc['Overall', 'Poisson Dev.'] = mean_poisson_deviance(*args)
    logger.info('Accuracy scores\n' + str(acc_df))

    return acc_df
