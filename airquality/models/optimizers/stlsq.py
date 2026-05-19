import numpy as np

from .base import BaseOptimizer

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class STLSQ(BaseOptimizer):
    """ Mixin class to provide STLSQ fitting functionality for an arbitrary fitter
        Assumes that each class will provide a _regress function 
    """
    def __init__(self, threshold=0., max_iter=20, fit_intercept=False, **kwargs):
        self.threshold = threshold
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        super().__init__(**kwargs)

    def _reduce(self, x, y, **fit_kwargs):
        """ Fit using sequentially thresholded least squares algorithm
        """
        N = x.shape[1]
        M = self.coef_.shape[0] - N
        logger.info(f"Applying sequentially-threshold least squares for {self.max_iter} iterations")
        for ii in range(self.max_iter):
            # Regress on current subset of coefficients
            num_nonzero = self.indices_.sum()
            if num_nonzero == 0:
                break
            logger.info(f'Iteration {ii}: {num_nonzero} / {len(self.indices_)} nonzero coefs')
            
            coefs = self.backend._regress(
                x[:, self.indices_[:N]], y, 
                start_params=self.coef_[self.indices_],
                **fit_kwargs)

            # Threshold coefficients following regression
            self.coef_[self.indices_] = np.where(np.abs(coefs) >= self.threshold, coefs, 0)
            self.indices_ = np.abs(self.coef_) >= self.threshold
            if M > 0:
                self.coef_[-M:] = coefs[-M:]  # Spline coefficients
                self.indices_[-M:] = True # Don't mask spline coefficients
                
            if self.indices_.sum() == num_nonzero:
                logger.info(f'STLSQ terminated after {ii+1} iterations')
                break