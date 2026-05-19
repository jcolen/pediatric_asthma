import torch
import lightning as L
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from typing import Union

from .data_load import data_load
from .preprocessing.prepare_analysis import prepare_zcta_analysis_df

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AirQualityDataModule(L.LightningDataModule):
    def __init__(self, 
                 data_load_config : dict = {},
                 preprocessing_config : dict = {},
                 exog : list = [], 
                 endog : Union[list, str] = None, 
                 dummies : Union[list, str] = ['dayofweek', 'monthofyear'], 
                 drop_first : bool = True,
                 offset : Union[list, str] = 'population', 
                 date_column : Union[list, str] = 'date_delta',
                 scale_exog : bool = False,
                 center_exog : bool = False,
                 test_split : float = 0.5, 
                 k_fold : int = 3,
                 k_fold_index : int = 0,
                 random_seed : int = 42):
        """ 
            Parameters
            - data_load_config : A dictionary containing information for ..utils.data_load
            - preprocessing_config : A dictionary containing information for ..utils.preprocessing
            - exog : a list of columns in the dataframe as inputs to the model
            - endog: a column name as a prediction target
            - dummies: an optional list of columns to be one-hot encoded
            - offset: a column for multiplicative offsets. Default None means no offset
            - scale_inputs: Standardize exogeneous inputs with zero_mean unit variance
            - test_split: Relative size of the train/test split
            - k_fold: Number of folds for k-fold cross validation
            - k_fold_index: index of this cross validation trial
            - random_seed: Used for train/test splitting
        """
        super().__init__()

        self.data_load_config = data_load_config
        self.preprocessing_config = preprocessing_config

        self.exog = exog
        self.endog = endog
        self.dummies = dummies
        self.drop_first = drop_first
        self.offset = offset
        self.date_column = date_column
        self.scale_exog = scale_exog
        self.center_exog = center_exog

        if not isinstance(endog, list):
            self.endog = [self.endog] # So that we have shape [N, 1]
        if offset is not None and not isinstance(offset, list):
            self.offset = [self.offset] # So that we have shape [N, 1]
        if date_column is not None and not isinstance(date_column, list):
            self.date_column = [self.date_column] # So that we have shape [N, 1]

        # Dataset splitting kwargs
        self.test_split = test_split
        self.k_fold = k_fold
        self.k_fold_index = k_fold_index
        self.random_seed = random_seed

        self._has_setup_data = False

    
    @property
    def exog_features(self):
        """ Continuous variables in the exogenous set """
        return self.exog
    
    @property
    def num_exog_features(self):
        return len(self.exog_features)
    
    @property
    def indicator_features(self):
        """ Indicator variables in the exogenous set """
        return self.all_exog_labels.drop(self.exog_features)
    
    @property
    def num_indicator_features(self):
        return len(self.indicator_features)

    @property
    def dataframe(self):
        """ The dataframe (dropped NaNs) corresponding to x, y, scale """
        return self.df.loc[self.index].reset_index(names='DataModule_index')

    @property
    def train_dataframe(self):
        return self.dataframe.iloc[self.train_ilocs]

    @property
    def test_dataframe(self):
        return self.dataframe.iloc[self.test_ilocs]

    @property
    def zctas(self):
        """ The unique zip codes contained in the dataset """
        return self.dataframe['ZCTA5CE20'].unique()

    @property
    def shape(self):
        return self.x.shape, self.y.shape, self.offset.shape

    def get_feature_names(self):
        return list(self.all_exog_labels.values)
    
    def get_exog_features(self):
        return self.exog_features
    
    def _setup_data(self):
        """ Load data into memory and prepare it accordingly """
        logger.info('Starting DataModule setup')

        self.dataframes = data_load(self.data_load_config)
        self.df, self.iqr = prepare_zcta_analysis_df(**self.dataframes, **self.preprocessing_config)

        logger.info(f'Exogenous variables: {self.exog}')
        exog = self.df.loc[:, self.exog]

        if self.center_exog:
            logger.info('Centering exogenous variables with sklearn StandardScaler')
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler(with_std=False)
            exog = pd.DataFrame(
                self.scaler.fit_transform(exog),
                columns=exog.columns,
                index=exog.index,
            )

        if self.scale_exog: # TODO: do this after train/test split on train data only
            logger.info('Standardizing exogenous variables with sklearn StandardScaler')
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            exog = pd.DataFrame(
                self.scaler.fit_transform(exog),
                columns=exog.columns,
                index=exog.index
            )

        # One-hot encode dummy columns
        logger.info(f'One-hot encoding columns: {self.dummies}')
        for dummy_column in self.dummies:
            dummy = pd.get_dummies(self.df[dummy_column], dtype=float, drop_first=self.drop_first)
            exog = pd.concat([exog, dummy], axis=1)

        exog = exog.dropna()
        self.all_exog_labels = exog.columns

        logger.info(f'Endogenous variables: {self.endog}')
        endog = self.df.loc[exog.index, self.endog]

        if self.offset is None:
            logger.warning('Not using population offset. Model is predicting unnormalized population counts')
            offset = pd.DataFrame({'population': 0}, index=exog.index)
        else:
            logger.info('Using population offset')
            offset = np.log(self.df.loc[exog.index, self.offset])

        if self.date_column is None:
            logger.warning('Not using date column')
        else:
            self.date_delta = self.df.loc[exog.index, self.date_column].values.astype(np.float32)

        self.index = exog.index # Retain index for future info
        self.x = exog.values.astype(np.float32)
        self.y = endog.values.astype(np.float32)
        self.offset = offset.values.astype(np.float32)

        logger.info(f'Built exogeneous array of shape {self.x.shape} and dtype {self.x.dtype}')
        logger.info(f'Built endogenous array of shape {self.y.shape} and dtype {self.y.dtype}')
        logger.info(f'Built offset array of shape {self.offset.shape} and dtype {self.offset.dtype}')
        if hasattr(self, 'date_delta'):
            logger.info(f'Built date_delta array of shape {self.date_delta.shape} and dtype {self.date_delta.dtype}')
        
        self._has_setup_data = True

    def setup(self, stage : str = 'fit'):
        if not self._has_setup_data:
            self._setup_data()

        if stage == 'fit':
            # Apply train validation split
            ilocs = np.arange(len(self.index), dtype=int)
            kf = GroupKFold(n_splits=self.k_fold, shuffle=True, random_state=self.random_seed)
            for ii, (train_ilocs, test_ilocs) in enumerate(kf.split(ilocs, groups=self.df.loc[self.index, 'ZCTA5CE20'])):
                if ii == self.k_fold_index:
                    break
            
            self.train_ilocs = train_ilocs
            self.test_ilocs = test_ilocs
    
    def get_train_data(self):
        return {
            'x': self.x[self.train_ilocs],
            'y': self.y[self.train_ilocs],
            'offset': self.offset[self.train_ilocs, 0],
            'date_delta': self.date_delta[self.train_ilocs, 0] if hasattr(self, 'date_delta') else None
        }

    def get_test_data(self):
        return {
            'x': self.x[self.test_ilocs],
            'y': self.y[self.test_ilocs],
            'offset': self.offset[self.test_ilocs, 0],
            'date_delta': self.date_delta[self.test_ilocs, 0] if hasattr(self, 'date_delta') else None
        }

    def get_data(self):
        return {
            'x': self.x,
            'y': self.y,
            'offset': self.offset[:, 0],
            'date_delta': self.date_delta[:, 0] if hasattr(self, 'date_delta') else None,
        }
