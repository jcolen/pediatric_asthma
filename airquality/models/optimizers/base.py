import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseOptimizer(BaseEstimator, RegressorMixin):
    """ Base class for sklearn optimization
    """
    def __init__(self, backend, normalize_columns : bool = False):
        self.backend = backend
        self.normalize_columns = normalize_columns
    
    def _reduce(self, x, y, **fit_kwargs):
        """ Apply the sparsification algorithm (e.g. STLSQ) if applicable
        """
        logger.info("No sparsification reduction implemented for " + __class__.__name__)
        return self

    @property
    def coef_(self):
        return self.backend.coef_
    
    def set_coef(self, coefs, indices):
        self.backend.set_coef(coefs, indices)
    
    def rescale_model(self, norms):
        self.backend.rescale_model(norms)
    
    def fit(self, x, y, **fit_kwargs):
        self.input_features_ = x.shape[1]
        self.output_features_ = y.shape[1]
        self.n_samples_ = x.shape[0]

        assert x.shape[0] == y.shape[0]
        assert self.output_features_ == 1

        x_in = np.copy(x)
        if self.normalize_columns:
            norms = np.linalg.norm(x_in, 2, axis=0)
            norms[norms == 0] = 1.
            x_in = x_in / norms

        # Make initial guess with model's base fit function
        self.backend.fit(x_in, y[:, 0], **fit_kwargs)
        self.indices_ = np.ones(self.coef_.shape, dtype=bool)

        # Apply reduction
        self._reduce(x_in, y[:, 0], **fit_kwargs)

        # Get coefficient support
        self.indices_ = np.abs(self.coef_) > 1e-10

        self.set_coef(self.coef_, self.indices_)
        if self.normalize_columns:
            self.rescale_model(norms)

        return self

    def predict(self, x, **kwargs):
        return self.backend.predict(x, **kwargs).reshape([-1, self.output_features_])

