import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from typing import Literal
_aggregate_types = Literal['daily', 'weekly']

from .threshold_dataframe import apply_absolute_threshold, apply_quantile_threshold, apply_stratification

def prepare_outcomes(patient_df : pd.DataFrame, 
                     zcta_df : pd.DataFrame, 
                     ICD10_code : str = 'J45', 
                     threshold : int = 100,
                     fill_zeros : bool = True,
                     stratify_outcomes : dict = None):
    logger.info(f'Restricting to ICD10 code {ICD10_code}')
    acute_counts = patient_df[patient_df['QUALIFYING_ICD_10_CODE'].str.startswith(ICD10_code)]
    logger.info(f'\tDataset has N = {len(acute_counts)}')

    if stratify_outcomes is not None:
        acute_counts = apply_stratification(acute_counts, **stratify_outcomes)

    logger.info(f'Grouping by date and ZCTA')
    acute_counts = acute_counts.groupby(['date_local', 'ZCTA5CE20']).size().to_frame('visits')
    logger.info(f'\tDataset has N = {len(acute_counts)}')

    # Restrict to relevant ZIP codes in Hampton Roads with patient visits
    logger.info(f'Restricting to Zip codes with >{threshold} in Hampton Roads region')
    zip_codes = acute_counts.groupby('ZCTA5CE20').agg({'visits': 'sum'})
    zip_codes = zip_codes[zip_codes['visits'] >= threshold].merge(zcta_df, how='inner', on='ZCTA5CE20').index
    outcomes = acute_counts[acute_counts.index.get_level_values(1).isin(zip_codes)]
    logger.info(f'\tDataset has N = {len(outcomes)}')

    # Now build filled-out multi-index to include zero-entries
    if fill_zeros:
        logger.info(f'Filling zeros for {zip_codes.nunique()} unique zip codes')
        full_index = pd.MultiIndex.from_product([outcomes.index.get_level_values(0).unique(), zip_codes.unique()], names=['date_local', 'ZCTA5CE20'])
        outcomes = outcomes.reindex(full_index, fill_value=0)
        logger.info(f'\tOutcomes dataset has N = {len(outcomes)}')

    return outcomes

def prepare_exposures(air_df : pd.DataFrame,
                      weather_df : pd.DataFrame,
                      covid_df : pd.DataFrame, 
                      svi_df : pd.DataFrame,
                      coi_df : pd.DataFrame,
                      rolling : bool = True,
                      lag : int = 0,
                      stratify_exposures : dict = None,
                      absolute_threshold : dict = None,
                      quantile_threshold : dict = None):
    # Add air quality measures and weather metrics
    logger.info(f'Adding air quality measures')
    logger.info(f'Adding weather metrics')
    exposures = air_df.merge(weather_df, on=['date_local', 'ZCTA5CE20'], how='inner')

    if lag > 0:
        if rolling:
            logger.info(f'Computing a rolling average over size {lag}')
            exposures = exposures.groupby('ZCTA5CE20').transform(
                lambda x: x.rolling(window=1+lag, center=False, min_periods=1).mean()
            ) # 1 + lag to include the current date as well
        else:
            logger.info(f'Applying a time lag of {lag} days to exposures')
            exposures = exposures.reset_index()
            exposures['date_local'] = exposures['date_local'].dt.to_timestamp() + pd.DateOffset(days=lag)
            exposures['date_local'] = exposures['date_local'].dt.to_period('D')
            exposures = exposures.set_index(['date_local', 'ZCTA5CE20'])

    # Covid data
    logger.info(f'Adding COVID data')
    exposures = exposures.join(covid_df, on='date_local', how='left')
    exposures['positive_covid_tests'] = exposures['positive_covid_tests'].fillna(0.)

    # Add SVI metrics
    logger.info(f'Adding Social Vulnerability index')
    exposures = exposures.join(svi_df, on='ZCTA5CE20', how='left')

    # Add COI metrics for each year
    logger.info(f'Adding Child Opportunity index')
    exposures['year'] = exposures.index.get_level_values('date_local').to_timestamp().year
    exposures = exposures.join(coi_df, on=['year', 'ZCTA5CE20'], how='left').drop(columns='year')

    if quantile_threshold is not None:
        exposures = apply_quantile_threshold(exposures, **quantile_threshold)

    if absolute_threshold is not None:
        exposures = apply_absolute_threshold(exposures, **absolute_threshold)
    
    if stratify_exposures is not None:
        exposures = apply_stratification(exposures, **stratify_exposures)

    return exposures

def weekly_aggregate(df, sum_columns=['visits', 'positive_covid_tests', 'TotalPrecipitation']):
    """ Weekly total visits and site-specific average air quality
        NOTE: Weekly aggregation has to occur AFTER merge because of overlap issues on left/right ends
    """
    logger.info(f'Aggregating data over 1-week periods')
    agg = {}
    for col in df.columns:
        if col in sum_columns:
            agg[col] = 'sum'
        else:
            agg[col] = 'mean'
    df = df.reset_index() \
        .assign(date_local=lambda x: x['date_local'].dt.to_timestamp().dt.to_period('W')) \
        .groupby(['date_local', 'ZCTA5CE20']) \
        .agg(agg)
    return df

def scale_units(df, 
                quantiles : list = [0.25, 0.75], 
                fixed_units : dict = {'visits': 1, 'Smoke': 1, 'positive_covid_tests': 1}):
    """ Apply a unit scaling for each parameter
        Some units can be defined manually, others default to the IQR
    """
    logger.info('Applying unit fixed and IQR scaling')
    iqr = df.quantile(quantiles[1]) - df.quantile(quantiles[0]) # Unit scale for each parameter
    for key in fixed_units:
        iqr[key] = fixed_units[key]
    iqr[['visits', 'positive_covid_tests', 'Smoke']] = 1.
    df = df / iqr
    df = df.reset_index()

    return df, iqr

def add_indicators(df):
    """ Add indicator variables for day of week, month of year, and a temporal offset
    """
    # Time delta from initial condition
    df['date_delta'] = df['date_local'].dt.to_timestamp()
    df['date_delta'] = (df['date_delta'] - df['date_delta'].min()).dt.days

    # Day of week indicator
    df['dayofweek'] = df['date_local'].dt.to_timestamp().dt.day_name()

    # Month indicator
    df['monthofyear'] = df['date_local'].dt.to_timestamp().dt.month_name()

    # Year indicator
    df['year'] = df['date_local'].dt.to_timestamp().dt.year.astype(str)

    return df

def prepare_zcta_analysis_df(patient_df : pd.DataFrame,
                             air_df : pd.DataFrame, 
                             weather_df : pd.DataFrame, 
                             svi_df : pd.DataFrame,
                             coi_df : pd.DataFrame,
                             zcta_df : pd.DataFrame, 
                             covid_df : pd.DataFrame, 
                             ICD10_code : str = '', 
                             aggregate : _aggregate_types ='daily', 
                             rolling : bool = False, 
                             lag : int = 0,
                             threshold : int = 100, 
                             fill_zeros: bool =True, 
                             fixed_units : dict = {'visits': 1, 'Smoke': 1, 'positive_covid_tests': 1},
                             stratify_outcomes : dict = None,
                             stratify_exposures : dict = None,
                             absolute_threshold : dict = None,
                             quantile_threshold : dict = None):
    """ Create a combined dataframe with all of the relevant information
        Parameters
            - patient_df : Patient visit information
            - air_df: Air quality information. Index = [date_local, ZCTA5CE20]
            - weather_df: Weather measures (temperature, etc.). Index = [date_local, ZCTA5CE20]
            - svi_df: Social vulnerability index dataframe. Index = ZCTA5CE20
            - coi_df: Child opportunity index dataframe. Index = [year, ZCTA5CE20]
            - zcta_df: Populations for each ZCTA. Index = ZCTA5CE20
            - covid_df: Covid positive test results. Index = date_local
            - ICD10_code: Starting letters/numbers for ICD10 codes of interest
            - aggregate: Daily or weekly aggregation of information
            - rolling: Whether to use rolling window aggregation (primarily with daily)
            - lag: Whether to include lagged continuous exposures
            - threshold: Restrict to zip codes with at least this number of total visits
            - fill_zeros: Label days not present in patient_df as visits=0 days
            - absolute_threshold: A threshold to apply to the combined df to restrict analysis window
            - quantile_threshold: A threshold to apply to the combined df to restrict analysis window
    """
    logger.info(f'START: Patient dataset has N = {len(patient_df)}')

    outcomes = prepare_outcomes(patient_df, 
                                zcta_df, 
                                ICD10_code=ICD10_code, 
                                threshold=threshold,
                                fill_zeros=fill_zeros,
                                stratify_outcomes=stratify_outcomes)
    
    exposures = prepare_exposures(air_df,
                                  weather_df,
                                  covid_df,
                                  svi_df,
                                  coi_df,
                                  rolling=rolling,
                                  lag=lag,
                                  stratify_exposures=stratify_exposures,
                                  quantile_threshold=quantile_threshold,
                                  absolute_threshold=absolute_threshold)
    combined_df = exposures.merge(outcomes, left_index=True, right_index=True, how='inner')

    if aggregate == 'weekly':
        combined_df = weekly_aggregate(combined_df)
    else:
        logger.info('Using daily exposure and outcome measurements')

    logger.info(f'\tDataset has N = {len(combined_df)}')

    combined_df, iqr = scale_units(combined_df, quantiles=[0.25, 0.75], fixed_units=fixed_units)

    combined_df = add_indicators(combined_df)

    # Merge local population as offset (remember to include constant)
    if not 'population' in combined_df.columns:
        logger.info('Adding population from Census data')
        combined_df = combined_df.merge(
            zcta_df['Population 0-19'],
            on='ZCTA5CE20',
            how='left'
        ).rename(columns={'Population 0-19': 'population'})

    # Include log-covid tests
    combined_df['log_covid'] = np.log1p(combined_df['positive_covid_tests'])

    return combined_df, iqr