# -*- coding: utf-8 -*-
"""
Code for the paper
"A churn prediction dataset from the telecom sector: a new benchmark for uplift modeling"
Anonymous authors
"""

from math import floor
from copy import deepcopy
import numpy as np
from tqdm.autonotebook import tqdm
from functions.easy_ensemble import EasyEnsemble


def generate_splits(N, k_folds, n_repeats, rng):
    res = []
    fold_size = N // k_folds
    for i in range(n_repeats):
        I = rng.permutation(N)
        for j in range(k_folds):
            res.append(I[j * fold_size : min((j + 1) * fold_size, N)])
    return res

def get_splits_from_results(past_results):
    return list([split["test_indices"] for split in past_results])

def benchmark(dataset, models, k_folds=5, n_repeats=4, seed=None, verbose=False, fit_params={}, predict_params={}, past_results=None):
    rng = np.random.default_rng(seed=seed)
    N = dataset.X.shape[0]
    all_indices = np.arange(N)
    if past_results is None:
        splits = generate_splits(N, k_folds, n_repeats, rng)
    else:
        splits = get_splits_from_results(past_results)
    n_splits = len(splits)
    
    res = []
    
    for i, split in tqdm(enumerate(splits), total=n_splits):
        if verbose:
            print("Split {}/{}".format(i, n_splits))
                  
        data_test = dataset[split]
        # Shuffle also the train indices just to be sure
        train_indices = rng.permutation(np.setdiff1d(all_indices, split))
        data_train = dataset[train_indices]
        
        res_split = {"test_indices": split, "results": {}}
        
        for model_name, model in models.items():
            model = deepcopy(model)
            if verbose:
                print("Fitting", model_name)
            model.fit(data_train, **fit_params.get(model_name, {}))
            if verbose:
                print("Predicting", model_name)
            pred = model.predict(data_test, **predict_params.get(model_name, {}))
            
            res_split["results"][model_name] = {}
            res_split["results"][model_name]["pred"] = pred
            res_split["results"][model_name]["model"] = model
        res.append(res_split)
    return res
