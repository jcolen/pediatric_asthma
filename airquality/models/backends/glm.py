import numpy as np
import pandas as pd

from io import StringIO
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.api import GLM
from statsmodels.genmod.families.family import Poisson

from .base import ModelWrapper

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GeneralizedLinearModel(BaseEstimator, RegressorMixin, ModelWrapper):
    """ Sklearn wrapper from statsmodels GLM
    """
    def __init__(self, 
                 scale : str = 'dev',
                 regularized : bool = False,
                 regularized_kwargs : dict = {}):
        self.scale = scale
        self.regularized = regularized
        self.regularized_kwargs = regularized_kwargs

    @property
    def coef_(self):
        return self.model_.params

    def set_coef(self, coefs, indices=None):
        self.model_.params[:] = coefs
        if indices is not None:
            self.model_.params[~indices] = 0.

            normalized_cov_params = self.model_.normalized_cov_params
            normalized_cov_params[:, ~indices] = 0.
            normalized_cov_params[~indices, :] = 0.
            self.model_.normalized_cov_params[:] = normalized_cov_params
    
    def rescale_model(self, norms):
        self.model_.params[:] = self.model_.params / norms

        normalized_cov_params = self.model_.normalized_cov_params
        normalized_cov_params /= norms[:, None]
        normalized_cov_params /= norms[None, :]
        self.model_.normalized_cov_params[:] = normalized_cov_params
    
    def _regress_model(self, x, y, offset=None, start_params=None, **kwargs):
        model = GLM(
            endog=y, 
            exog=x, 
            offset=offset,
            family=Poisson()
        )
        if self.regularized:
            return model.fit_regularized(start_params=start_params, **self.regularized_kwargs)
        else:
            return model.fit(scale=self.scale, start_params=start_params)        

    def _regress(self, x, y, offset=None, **kwargs):
        coef = (
            self._regress_model(x, y, offset=offset, **kwargs)
            .params
        )
        return coef

    def fit(self, x, y, offset=None, **kwargs):
        # Populate initial guess using basic fit method
        self.model_ = self._regress_model(x, y, offset=offset, **kwargs)
        return self

    def predict(self, x, offset=None, **kwargs):
        return self.model_.predict(
            exog=x, 
            offset=offset
        )
    
    def summary_dataframe(self, feature_names):
        summary = self.model_.summary(xname=feature_names)
        csv_data = StringIO(summary.tables[1].as_csv().replace(" ", ""))
        summary_df = pd.read_csv(csv_data, index_col=0)

        summary_df['risk_ratio'] = np.exp(summary_df['coef'])
        summary_df['lower_ci'] = np.exp(summary_df['[0.025'])
        summary_df['upper_ci'] = np.exp(summary_df['0.975]'])
        return summary_df.rename_axis('parameter')