import numpy as np
from typing import Callable
from copy import deepcopy

from .base import BaseOptimizer

class EnsembleOptimizer(BaseOptimizer):
    """ A meta-optimizer that averages repeated trials over multiple models
        Includes:
            bagging - boostrap aggregation of many models trained 
                on random subsets of input data
            library ensembling - aggregation of models trained
                with random subsets of library terms
        This is implemented roughly following pysindy.optimizers.EnsembleOptimizer
        but has to be modified to allow the unbias step to occur using any optimizer,
        not just LinearRegression
    """
    def __init__(self,
                 base : BaseOptimizer,
                 bagging : bool = False,
                 library_ensemble : bool = False,
                 n_models : int = 20,
                 subset_size : float = 0.5,
                 n_candidates_to_drop: int = 1,
                 replace : bool = True,
                 ensemble_aggregator : Callable = None):
        super().__init__(backend=deepcopy(base.backend))
        self.base = base
        self.bagging = bagging
        self.library_ensemble = library_ensemble
        self.n_models = n_models
        self.subset_size = subset_size
        self.n_candidates_to_drop = n_candidates_to_drop
        self.replace = replace
        self.ensemble_aggregator = ensemble_aggregator

        self.coef_list = [] # Stores list of fitted models

    def _reduce(self, x, y, **fit_kwargs):
        """ Repeatedly fit self.base on subsets of data and libraries
        """
        n_subset = min(int(self.subset_size * self.n_samples_), self.n_samples_)
        n_candidates = max(1, self.input_features_ - self.n_candidates_to_drop)
        n_extra_features = len(self.coef_) - self.input_features_

        for ii in range(self.n_models):
            if self.bagging:
                # Sample dataset for this trial
                idxs = np.random.choice(range(self.n_samples_), n_subset, replace=self.replace)
                x_subset = x[idxs, :]
                y_subset = y[idxs]
                subset_kwargs = { k : (v[idxs] if isinstance(v, np.ndarray) \
                                       and v.shape[0] == self.n_samples_ else v) \
                                 for k, v in fit_kwargs.items()}
            else:
                x_subset, y_subset = x, y
                subset_kwargs = fit_kwargs
            
            lib_idxs = np.arange(self.input_features_, dtype=int)
            if self.library_ensemble:
                # Sample library elements for this trial
                lib_idxs = np.random.choice(lib_idxs, n_candidates, replace=False)
                lib_idxs = np.sort(lib_idxs)
                x_subset = x_subset[:, lib_idxs]
            
            # Fit on the subset
            self.base.fit(x_subset, y_subset[:, None], **subset_kwargs)
            subset_coefs = np.zeros(self.indices_.shape)
            subset_coefs[:self.input_features_][lib_idxs] = self.base.coef_[:-n_extra_features]
            subset_coefs[-n_extra_features:] = self.base.coef_[-n_extra_features:]
            self.coef_list.append(subset_coefs)
        
        # Aggregate coefficients over trials
        if self.ensemble_aggregator is None:
            self.set_coef(np.median(self.coef_list, axis=0))
        else:
            self.set_coef(self.ensemble_aggregator(self.coef_list))