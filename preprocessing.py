# -*- coding: utf-8 -*-
import numpy as np
from helpers import *

#Globalement fonctionne parfaitement, seul question niveau interpretation plutôt :
#Le sample est basé sur shuffle, sauf que du coup ça shuffle les temps et les positions, ça change rien pour le NN,
#mais peut être pour interpretation pas cool

'''
Preprocessing
'''

###########Sampling methods##########

def sample_points(ux, uy, ratio, seed=1) :
    '''
    This function samples 1 datapoint which is contained in the matrices ux and uy (matrices obtained by a single simulation).
    The sampling occurs along the axis 0 which corresponds to the positions of the points on the grid of the simulation.
    Inputs:
        * ux : matrix of the velocity along x for different positions (axis 0) and different time steps (axis 1)
        * uy : matrix of the velocity along y for different positions (axis 0) and different time steps (axis 1)
        * ratio : # of positions sampled = ratio * # of positions initially
        * seed : to generate the random for sampling
    Outputs:
        * new_ux : Basically same matrice as ux but with rows (positions) sampled (and shuffled, due to the way the sampling is made) 
        * new_uy : Basically same matrice as uy but with rows (positions) sampled (and shuffled, due to the way the sampling is made) 
    '''
    # set seed
    np.random.seed(seed)
    # generate random indices
    indices = np.random.permutation(ux.shape[0])
    new_inds = indices[ : int(np.floor(ratio * ux.shape[0]))]
    new_ux = ux[new_inds, :]
    new_uy = uy[new_inds, :]
    
    return new_ux, new_uy


def sample_times(ux, uy, ratio, seed=1) :
    '''
    This function samples 1 datapoint which is contained in the matrices ux and uy (matrices obtained by a single simulation).
    The sampling occurs along the axis 1 which corresponds to the time steps of the simulation.
    Inputs:
        * ux : matrix of the velocity along x for different positions (axis 0) and different time steps (axis 1)
        * uy : matrix of the velocity along y for different positions (axis 0) and different time steps (axis 1)
        * ratios : # of time steps sampled = ratio * # of time steps initially
        * seed : to generate the random for sampling
    Outputs:
        * new_ux : Basically same matrice as ux but with columns (times) sampled (and shuffled, due to the way the sampling is made) 
        * new_uy : Basically same matrice as uy but with columns (times) sampled (and shuffled, due to the way the sampling is made) 
    '''
    # set seed
    np.random.seed(seed)
    # generate random indices
    indices = np.random.permutation(ux.shape[1])
    new_inds = indices[ : int(np.floor(ratio * ux.shape[1]))]
    new_ux = ux[:, new_inds]
    new_uy = uy[:, new_inds]
    
    return new_ux, new_uy


def get_point(U, n, size) :
    '''
    Get datapoint number n from the matrix Ux or Uy (recall that 1 datapoint is the output matrix of 1 Stokes simulation)
    Inputs
        * U : Matrix of velocities at different position on the grid and at different time step (2D matrix)
                # of rows : nb_of_datapoints * nb_rows_of_1_datapoint
                # of columns : nb_of_time_steps of the simulations
        * n : number of data point of interest
        * size : # of rows (points) belonging to 1 simulation, in our case 1 simulation = 1 datapoint for our neural networks
    Outputs:
        * U_n : Datapoint number n from the matrix U
                # of rows : nb_rows_of_1_datapoint
                # of columns : nb_of_time_steps of the simulations
    '''
    U_n = U[n*size:(n+1)*size, :]
    return U_n


def sample(Ux, Uy, ratio_pts, ratio_t, size=5509) : #Checker size !!!!!!!!
    '''
    Function that samples the original matrices Ux and Uy along the axis 0 (samples the number of points considered on the grid
    of the simulation), and over the axis 1 (samples the number of time steps considered)
    Inputs
        * Ux : Matrix of velocities along x at different position on the grid and at different time step (2D matrix)
        * Uy : Matrix of velocities along y at different position on the grid and at different time step (2D matrix)
        * ratio_pts : # of positions sampled = ratio* # of positions initially
        * ratios_t : # of time_steps sampled = ratio* # of time_steps initially
        * size : # of rows (points) belonging to 1 simulation, in our case 1 simulation = 1 datapoint for our neural networks
    Outputs:
        * new_Ux : Basically same matrice as ux but with rows (positions) and columns (times) sampled (and shuffled, due to the way the 
                    sampling is made) 
        * new_Uy : Basically same matrice as uy but with rows (positions) and columns (times) sampled (and shuffled, due to the way the 
                    sampling is made) 
    '''
    size = int(np.floor(size))
    new_Ux = []
    new_Uy = []
    for i in range(int(Ux.shape[0]/size)) :
        ux = get_point(Ux, i, size)
        uy = get_point(Uy, i, size)
        
        ux_s1, uy_s1 = sample_points(ux, uy, ratio_pts)
        ux_s2, uy_s2 = sample_times(ux_s1, uy_s1, ratio_t)
        
        new_Ux.append(ux_s2)
        new_Uy.append(uy_s2)  
        
    new_Ux = np.concatenate(new_Ux)
    new_Uy = np.concatenate(new_Uy)
    
    return new_Ux, new_Uy
        
#ATTENTION : marche pas, nen pas utiliser pour l'instant
def get_samples(Ux, Uy, ratios_pts, ratios_t, size=5509) :
    
    ratios_pts.insert(0, 1)
    ratios_t.insert(0, 1)
    
    #Idée : plutot que liste, faire tableau 2D et referer à son voisin direct de gauche si il existe, sinon son voisin du dessus
    samples_Ux = [Ux]
    samples_Uy = [Uy]
    
    for i in range(1, len(ratios_pts)) :
        for j in range(1, len(ratios_t)) :
            new_Ux, new_Uy = sample(samples_Ux, samples_Uy, ratios_pts[i], ratios_t[i], size=samples_Ux.shape[0]*ratio_pts[i-1])
            samples_Ux.append(new_Ux)
            samples_Uy.append(new_Uy)
    return 0 


###########Flattening methods##########

def flatten(Ux, Uy, ratios_pts, size=5509):
    '''
    Function that flattens the original matrices Ux and Uy into a final array of dimension (2*Nu*Nt) * Ns
    Inputs:
        * Ux : Matrix of velocities along x at different position on the grid and at different time step (2D matrix)
        * Uy : Matrix of velocities along y at different position on the grid and at different time step (2D matrix)
        * ratio_pts : # of positions sampled = ratio* # of positions initially
        * size : # of rows (points) belonging to 1 simulation, in our case 1 simulation = 1 datapoint for our neural networks
    Outputs:
        * flattened_array_all : final flattened array to dimension (2*Nu*Nt) * Ns
    '''
    
    # defining index which separetes each simulation with respect to the sampling
    idx_new_sim = np.int(np.floor(ratios_pts * size))
    
    # defining the arrays where we will seperate the different simulation
    simulation_x = []
    simulation_y = []
    
    # for loop to separate the Ux and Uy simulations 
    j = 0
    for i in range(np.int(Ux.shape[0]/idx_new_sim)):
        simulation_x.append([Ux[j:j+idx_new_sim,:]])
        simulation_y.append([Ux[j:j+idx_new_sim,:]])
        j = j + idx_new_sim
    simulation_x = np.array(simulation_x).squeeze()
    simulation_y = np.array(simulation_y).squeeze()
    
    # initializing the output flattened array
    flattened_array_all = np.zeros(2*simulation_x.shape[1]*simulation_x.shape[2])
    
    # for loop to iterate through the simulations
    for idx_sim in range(simulation_x.shape[0]):
        sim_x = simulation_x[idx_sim]
        sim_y = simulation_y[idx_sim]
        flattened_array = np.array([])
        # for loop to iterate through the time steps
        for col in range(simulation_x.shape[2]):
            # get successively the Ux(t) then Uy(t) at a fixed time step t 
            flattened_array = np.append(flattened_array, sim_x[:, col])
            flattened_array = np.append(flattened_array, sim_y[:, col])
        # add all the time steps to our final flattened array  
        flattened_array_all = np.c_[flattened_array_all, flattened_array]
    
    # delete the initialization of zeros in the first column
    flattened_array_all = np.delete(flattened_array_all, 0, axis=1)
    
    return flattened_array_all.T


