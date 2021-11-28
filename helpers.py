# -*- coding: utf-8 -*-
"""
Helpers functions
"""
import numpy as np


def load_csv_data(data_path):
    """Loads data and returns X (which is also Y in our case) as a numpy array"""
    x = np.genfromtxt(data_path, delimiter=",")

    return x

def get_Nu_Nt_sampled(U, U_sampled) :
    """
    Get the number of rows in the new U matrix after sampling
    Inputs :
        U : Original matrix U
            # of rows : nb_of_datapoints * nb_rows_of_1_datapoint_before_sampling = nb_of_datapoints * 5509 
            # of columns : nb_of_time_steps_before_sampling
        U_sampled : Sampled version of the matrix U
            # of rows : nb_of_datapoints * nb_rows_of_1_datapoint_after_sampling =  nb_of_datapoints * 5509 * ratio_pts
            # of columns : nb_of_time_steps_after_sampling = nb_of_time_steps_before_sampling * ratio_times
        (Recall : ratio_times/ratio_pts = ratio of times/positions that we keep in U_sampled)
    Outputs :
        Nu_sampled : nb_rows_of_1_datapoint_after_sampling = 5509 * ratio_pts
    """

    nb_of_pts = U.shape[0]/5509
    
    return int(U_sampled.shape[0]/nb_of_pts), int(U.shape[1])
