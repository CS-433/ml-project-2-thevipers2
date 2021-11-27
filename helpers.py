# -*- coding: utf-8 -*-
"""
Helpers functions
"""
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns X (which is also Y in our case) as a numpy array"""
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 1:]

    return input_data, ids