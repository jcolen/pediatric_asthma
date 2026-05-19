import numpy as np
import pandas as pd
import warnings

from scipy.linalg import LinAlgWarning
from sklearn.linear_model import PoissonRegressor

from .base import ModelWrapper

class Poisson(PoissonRegressor, ModelWrapper):
    """ Poisson Regression wrapped up in our sklearn wrapper
        You may say this is unnecessary, but it provides a template for 
    """
    def _regress(self, x, y, offset=None, **kwargs):
        """ Perform Poisson regression
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=LinAlgWarning)
            try:
                coef = (
                    PoissonRegressor(**self.get_params())
                    .fit(x, y / np.exp(offset))
                    .coef_
                )
            except LinAlgWarning:
                self.alpha = 2 * self.alpha
        return coef

    def fit(self, x, y, offset=None, **kwargs):
        if offset is None:
            return super().fit(x, y)
        
        return super().fit(x, y / np.exp(offset))
    
    def set_coef(self, coefs, indices=None):
        self.coef_[:] = coefs
    
    def predict(self, x, offset=None, **kwargs):
        """ For interoperatbility we return [n_samples, n_features]
        """
        pred = super().predict(x)[:, None]
        if offset is None:
            return pred
        return pred * np.exp(offset)[:, None]
    
    def summary_dataframe(self, feature_names):
        summary_df = pd.DataFrame({
            'parameter': feature_names,
            'coef': self.coef_,
        }).set_index('parameter')
        return summary_df