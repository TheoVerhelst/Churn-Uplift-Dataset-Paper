# -*- coding: utf-8 -*-
"""
Code for the paper
"A churn prediction dataset from the telecom sector: a new benchmark for uplift modeling"
Anonymous authors
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from causalml.inference.tree import UpliftRandomForestClassifier

class RandomForestWrapper:
    def __init__(self, *args, **kwargs):
        self.model = RandomForestClassifier(*args, **kwargs)
    
    def fit(self, data, *args, **kwargs):
        # Train on control group only
        return self.model.fit(data.X[~data.t], data.y[~data.t], *args, **kwargs)
    
    def predict(self, data, *args, **kwargs):
        return self.model.predict_proba(data.X, *args, **kwargs)[:, 1]

class SLearnerWrapper:
    def __init__(self, *args, **kwargs):
        self.model_0 = RandomForestClassifier(*args, **kwargs)
        self.model_1 = RandomForestClassifier(*args, **kwargs)
    
    def fit(self, data, *args, **kwargs):
        return (
            self.model_0.fit(data.X[~data.t], data.y[~data.t], *args, **kwargs),
            self.model_1.fit(data.X[data.t], data.y[data.t], *args, **kwargs)
        )
    
    def predict(self, data, *args, **kwargs):
        return pd.DataFrame(data={
            "control": self.model_0.predict_proba(data.X, *args, **kwargs)[:, 1],
            "target": self.model_1.predict_proba(data.X, *args, **kwargs)[:, 1]
        })

class URFCWrapper:
    def __init__(self, *args, **kwargs):
        self.model = UpliftRandomForestClassifier(control_name="control", *args, **kwargs)
    
    def fit(self, data, *args, **kwargs):
        t = np.array(["target" if t else "control" for t in data.t])
        return self.model.fit(data.X, t, data.y, *args, **kwargs)
    
    def predict(self, data, *args, **kwargs):
        return self.model.predict(data.X, *args, **kwargs)

