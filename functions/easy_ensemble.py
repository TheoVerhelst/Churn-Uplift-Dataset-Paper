# -*- coding: utf-8 -*-
"""
Code for the paper
"A churn prediction dataset from the telecom sector: a new benchmark for uplift modeling"
Anonymous authors
"""

from functools import reduce
from copy import copy
from joblib import Parallel, delayed
import numpy as np
from numpy.random import default_rng

class EasyEnsemble:
    def __init__(self, base_model, n_folds, random_state=None, n_jobs = 1, verbose=False):
        self.base_model = base_model
        self.n_folds = n_folds
        self.rng = default_rng(seed=random_state)
        self.models = None
        self.verbose = verbose
        self.n_jobs = n_jobs
    
    def _compute_fold(self, i, indices_1, data, N_0, N_1, *args, **kwargs):
        indices_0 = self.rng.integers(low=0, high=N_0, size=N_1)
        indices = np.hstack((indices_1, indices_0))
        self.rng.shuffle(indices)
        data_i = data[indices]
        model = copy(self.base_model)
        model.fit(data_i, *args, **kwargs)
        return {"model": model, "training_indices": indices}
    
    def fit(self, data, *args, **kwargs):
        mask_1 = data.y == 1
        indices_1 = np.arange(mask_1.shape[0])[mask_1]
        N_0 = np.sum(data.y == 0)
        N_1 = np.sum(data.y == 1)
        if self.verbose:
            print("{} negative samples and {} positive samples".format(N_0, N_1))
        self.models = Parallel(
            n_jobs=self.n_jobs,
            verbose=10 if self.verbose else 0
        )(delayed(self._compute_fold)(i, indices_1, data, N_0, N_1, *args, **kwargs) for i in range(self.n_folds))
    
    def predict(self, data, *args, **kwargs):
        # Use a lambda to reduce the predictions.
        # We could use np.add but it wouldn't work with pandas DataFrames
        predictions = reduce(
            lambda a, b: a + b,
            Parallel(
                n_jobs=self.n_jobs,
                verbose=10 if self.verbose else 0
            )(delayed(model["model"].predict)(data, *args, **kwargs) for model in self.models)
        )
        return predictions / len(self.models)
