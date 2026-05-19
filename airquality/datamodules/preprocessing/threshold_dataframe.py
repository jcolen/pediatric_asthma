import pandas as pd

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def apply_stratification(df, column, values):
    if isinstance(values, list):
        logger.info(f'Applying bounds on {column} [{values[0]:.4g}, {values[1]:.4g})')
        logger.info(df[column].describe())
        df = df[(df[column] >= values[0]) & (df[column] < values[1])]
    elif isinstance(values, str):
        logger.info(f'Selecting rows where {column} = {values}')
        logger.info(df[column].unique())
        df = df[df[column] == values]

    logger.info(f'Selected dataframe has size {len(df)}')
    return df

def apply_absolute_threshold(df, column, threshold, keep='upper'):
    logger.info(f'Applying threshold on {column} with threshold {threshold:.4g}')

    if keep == 'upper':
        df = df[df[column] >= threshold]
    elif keep == 'lower':
        df = df[df[column] < threshold]
    else:
        raise ValueError(f'Keywork keep={keep} must be in [\'upper\', \'lower\']')

    logger.info(f'Thresholded dataframe has size {len(df)}')
    return df

def apply_quantile_threshold(df, column, quantile=0.5, keep='upper'):
    logger.info(f'Applying quantile threshold on {column} with quantile {quantile:.4g}')
    threshold = df[column].quantile(quantile)
    return apply_absolute_threshold(df, column, threshold, keep)

    
    