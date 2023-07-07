# -*- coding: utf-8 -*-
"""
Code for the paper
"A churn prediction dataset from the telecom sector: a new benchmark for uplift modeling"
Anonymous authors
"""

from sklearn.model_selection import train_test_split
import numpy as np

class Dataset:
    def __init__(self, **arrays):
        self.__dict__.update(arrays)
    
    def __getitem__(self, key):
        res = Dataset()
        res.__dict__.update(self.__dict__)
        for name in res.__dict__:
            res.__dict__[name] = res.__dict__[name][key].copy()
        return res
        #return Dataset(**{name: value[key] for name, value in self.__dict__.items()})
    
    def train_test_split(self, *args, **kwargs):
        ordered_dict = list(self.__dict__.items())
        res = train_test_split(*list(value for name, value in ordered_dict), *args, **kwargs)
        return (
            Dataset(**{ordered_dict[i][0]: res[i * 2] for i in range(len(ordered_dict))}),
            Dataset(**{ordered_dict[i][0]: res[i * 2 + 1] for i in range(len(ordered_dict))})
        )
