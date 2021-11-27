# -*- coding: utf-8 -*-
"""
Helpers functions
"""
import numpy as np


def load_csv_data(data_path):
    """Loads data and returns X (which is also Y in our case) as a numpy array"""
    x = np.genfromtxt(data_path, delimiter=",")

    return x