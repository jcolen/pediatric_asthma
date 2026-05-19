import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from . import feature_library as feature_libraries
from . import backends as backends
from . import optimizers as optimizers

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StationaryModel(BaseEstimator, RegressorMixin):
    """ Stationary models predict stationary counts, typically via Poisson regression
    """
    def __init__(self, 
                 datamodule = None,
                 feature_library : dict = None,
                 backend : dict = None,
                 optimizer : dict = None):
        self.feature_names = datamodule.get_feature_names() # For model printing
        self.exog_features = datamodule.get_exog_features() # Only summarize continuous non-indicator variables
        self.feature_library = feature_libraries.make(
            num_exog_features=datamodule.num_exog_features, 
            num_indicator_features=datamodule.num_indicator_features,
            **feature_library
        )
        self.backend = backends.make(**backend)
        self.optimizer = optimizers.make(
            backend=self.backend, 
            **optimizer
        )
    
    def get_feature_names(self):
        return self.feature_library.get_feature_names(input_features=self.feature_names)
    
    def fit(self, x, y, offset=None, date_delta=None, **kwargs):
        """ Stationary fitting proceeds like a typical sklearn optimizer
        """
        steps = [
            ("scaler", StandardScaler(with_mean=True, with_std=False)),
            ("features", self.feature_library),
            ("model", self.optimizer)
        ]
        self.model = Pipeline(steps)
        self.model.fit(x, y, model__offset=offset, model__date_delta=date_delta)

        return self

    def predict(self, x, **kwargs):
        return self.model.predict(x, **kwargs)
    
    def coefficients(self):
        return self.optimizer.coef_
    
    def print(self, lhs='y', precision=3):
        def term(coef, feature):
            rounded_coef = np.round(coef, precision)
            if rounded_coef == 0: 
                return ""
            return f"{coef:.{precision}f} {feature}"

        terms = [term(coef, feature) for coef, feature in 
                 zip(self.coefficients(), self.get_feature_names())]
        rhs = ' + '.join(filter(bool, terms))
        print(lhs + ' = ' + rhs)

    def estimate_risk_ratio(self, x, idx, **predict_kwargs):
        """ Compute outcome change due to a 1-unit exposure increase in input parameter idx """
        # Model coefficients are the only random variables with uncertainty to propagate
        model = self.model['model'].backend.model_
        mu_coefs = model.params[:len(self.get_feature_names())]
        cov_coefs = model.cov_params()[:len(self.get_feature_names()), :len(self.get_feature_names())]

        # Measurement values are not random variables but constant coefficients to the variables
        x = self.model['scaler'].transform(x)
        # x[:, idx] = x[:, idx].mean() # Replace exposure of interest with mean value
        baseline_features = self.model['features'].transform(x)

        x[:, idx] += 1. # Effect a 1-unit increase in exposure
        exposed_features = self.model['features'].transform(x)

        # Compute mean relative risk and error using error propagation formula
        exposed = np.exp(np.einsum('Ni,i->N', exposed_features, mu_coefs))
        baseline = np.exp(np.einsum('Ni,i->N', baseline_features, mu_coefs))

        exposed_sum = exposed.sum() 
        dE_di = np.einsum('Ni,N->i', exposed_features, exposed)
        d2E_dij = np.einsum('Ni,Nj,N->ij', exposed_features, exposed_features, exposed)

        baseline_sum = baseline.sum()
        dB_di = np.einsum('Ni,N->i', baseline_features, baseline)
        d2B_dij = np.einsum('Ni,Nj,N->ij', baseline_features, baseline_features, baseline)

        # Covariance correction to mean from error propagation
        cov_correction = (
            d2E_dij * baseline_sum**3 - d2B_dij * baseline_sum**2 * exposed_sum - \
            (dE_di[:,None] * dB_di[None,:] + dE_di[None,:] * dB_di[:,None]) * baseline_sum**2 + \
            2 * dB_di[:,None] * dB_di[None,:] * exposed_sum * baseline_sum
        ) / baseline_sum**4
        RR = exposed_sum / baseline_sum + 0.5 * np.einsum('ij,ij', cov_coefs, cov_correction)

        # Compute variance using the Delta method
        dRR_di = (dE_di * baseline_sum - dB_di * exposed_sum) / baseline_sum**2
        var_RR = np.einsum('i,j,ij', dRR_di, dRR_di, cov_coefs)

        # Compute 95% CI
        CI = 1.96 * np.sqrt(var_RR)
        lower_CI = RR - CI
        upper_CI = RR + CI

        return RR, lower_CI, upper_CI

    def summary(self, x, y=None, verbose=True, **predict_kwargs):

        # Get the summary dataframe from the model itself
        summary_df = self.optimizer.backend.summary_dataframe(feature_names=self.get_feature_names())
        
        # Add the estimated risk ratios
        for idx, parameter in enumerate(tqdm(self.exog_features)):
            rr, lower_ci, upper_ci = self.estimate_risk_ratio(x.copy(), idx, **predict_kwargs)
            summary_df.loc[parameter, 'risk_ratio (est.)'] = rr
            summary_df.loc[parameter, 'lower_ci (est.)'] = lower_ci
            summary_df.loc[parameter, 'upper_ci (est.)'] = upper_ci
        if verbose:
            self.print(precision=3)
            logger.info(f'Model has {np.count_nonzero(self.coefficients())} / {self.coefficients().size}'
                        ' non-zero coefficients')
            logger.info('\n' + str(summary_df.loc[self.exog_features]))
        return summary_df

