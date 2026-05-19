import numpy as np

from .base import BaseOptimizer
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import f_regression, mutual_info_regression

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SingleVariableSelection(BaseOptimizer):
    """ Class that performs variable selection as part of a pipeline before performing
        a more standard optimization routine
    """
    def __init__(self, p_threshold=0.05, **kwargs):
        self.p_threshold = p_threshold
        super().__init__(**kwargs)

    def _reduce(self, x, y, offset=None, **kwargs):
        """ 
        """
        logger.info(f"Using sklearn SelectFpr to select regression features with p < {self.p_threshold}")
        var_selector = SelectFpr(f_regression, alpha=self.p_threshold)
        if offset is None:
            var_selector.fit(x, y)
        else:
            logger.info("Selecting variables using logged population offset")
            var_selector.fit(x[:, 1:], y / np.exp(offset))
        
        self.indices_[1:] = var_selector.get_support()
        logger.info(f"Selected {self.indices_.sum()} / {len(self.indices_)} nonzero coefs")
        self.coef_[~self.indices_] = 0
        self.coef_[self.indices_] = self.backend._regress(
            x[:, self.indices_[:x.shape[1]]], y, 
            offset=offset, **kwargs)