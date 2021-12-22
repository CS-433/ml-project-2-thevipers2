# -*- coding: utf-8 -*-
import numpy as np
import _pickle as cPickle
from helpers import *

'''
Preprocessing of data.
'''

# -----------SAMPLING METHODS------------------

def sample_points(ux, uy, ratio, seed=1) :
    '''
    This function samples 1 datapoint which is contained in the matrices ux and uy (matrices obtained by a single simulation).
    The sampling occurs along the axis 0 which corresponds to the positions of the points on the grid of the simulation.
    Inputs:
        * ux (np.array): matrix of the velocity along x for different positions (axis 0) and different time steps (axis 1)
        * uy (np.array): matrix of the velocity along y for different positions (axis 0) and different time steps (axis 1)
        * ratio (float): # of positions sampled = ratio * # of positions initially
        * seed (int): to generate the random for sampling
    Outputs:
        * new_ux (np.array): same matrice as ux but with rows (positions) sampled (and shuffled, due to the way the sampling is made) 
        * new_uy (np.array): same matrice as uy but with rows (positions) sampled (and shuffled, due to the way the sampling is made) 
        * new_inds (list): new indices that were sampled
    '''
    # set seed
    np.random.seed(seed)
    # generate random indices
    indices = np.random.permutation(ux.shape[0])
    new_inds = indices[ : int(np.floor(ratio * ux.shape[0]))]
    
    #generate "position-sampled" ux and uy :
    new_ux = ux[new_inds, :]
    new_uy = uy[new_inds, :]
    
    return new_ux, new_uy, new_inds


def sample_times(ux, uy, ratio) :
    '''
    This function samples 1 datapoint which is contained in the matrices ux and uy (matrices obtained by a single simulation).
    The sampling occurs along the axis 1 which corresponds to the time steps of the simulation.
    The sampling is not uniform (equispaced)
    Inputs:
        * ux (np.array): matrix of the velocity along x for different positions (axis 0) and different time steps (axis 1)
        * uy (np.array): matrix of the velocity along y for different positions (axis 0) and different time steps (axis 1)
        * ratios (float): # of time steps sampled = ratio * # of time steps initially
        * seed (int): to generate the random for sampling
    Outputs:
        * new_ux (np.array): Basically same matrice as ux but with columns (times) sampled
        * new_uy (np.array): Basically same matrice as uy but with columns (times) sampled
    '''
    
    #step = #indices/#new_indices = N / ratio * N = 1/ratio
    step = int(1/ratio)
    # generate indices
    new_inds = np.arange(0, ux.shape[1], step)
    
    #generate "time-sampled" ux and uy :
    new_ux = ux[:, new_inds]
    new_uy = uy[:, new_inds]
    
    return new_ux, new_uy


def get_point(U, n, size) :
    '''
    Get datapoint number n from the matrix Ux or Uy (recall that 1 datapoint is the output matrix of 1 Stokes simulation)
    Inputs
        * U (np.array): matrix of velocities at different position on the grid and at different time step (2D matrix)
            # of rows : nb_of_datapoints * nb_rows_of_1_datapoint
            # of columns : nb_of_time_steps of the simulations
        * n (int): number of data point of interest
        * size (int): # of rows (points) belonging to 1 simulation, in our case 1 simulation = 1 datapoint for our neural networks
    Outputs:
        * U_n (np.array): datapoint number n from the matrix U
            # of rows : nb_rows_of_1_datapoint
            # of columns : nb_of_time_steps of the simulations
    '''
    U_n = U[n*size:(n+1)*size, :]
    return U_n


def sample(Ux, Uy, ratio_pts, ratio_t, size=5509) :
    '''
    Function that samples the original matrices Ux and Uy along the axis 0 (samples the number of points considered on the grid
    of the simulation), and over the axis 1 (samples the number of time steps considered)
    Inputs
        * Ux (np.array): Matrix of velocities along x at different position on the grid and at different time step (2D matrix)
        * Uy (np.array): Matrix of velocities along y at different position on the grid and at different time step (2D matrix)
        * ratio_pts (list): # of positions sampled = ratio* # of positions initially
        * ratios_t (list): # of time_steps sampled = ratio* # of time_steps initially
        * size (int): # of rows (points) belonging to 1 simulation, in our case 1 simulation = 1 datapoint for our neural networks
    Outputs:
        * new_Ux (np.array): Basically same matrice as ux but with rows (positions) and columns (times) sampled (and shuffled, due to the way the sampling is made) 
        * new_Uy (np.array): Basically same matrice as uy but with rows (positions) and columns (times) sampled (and shuffled, due to the way the sampling is made) 
        * new_inds (list): indices of the sampled positions (axis=0) : # new_inds = ratio_pts * initial number of positions = ratio_pts * size    
    '''
    size = int(np.floor(size))
    new_Ux = []
    new_Uy = []
    for i in range(int(Ux.shape[0]/size)) :
        ux = get_point(Ux, i, size)
        uy = get_point(Uy, i, size)
        
        ux_s1, uy_s1, new_inds = sample_points(ux, uy, ratio_pts)
        ux_s2, uy_s2 = sample_times(ux_s1, uy_s1, ratio_t)
        
        new_Ux.append(ux_s2)
        new_Uy.append(uy_s2)  
    
    new_Ux = np.concatenate(new_Ux)
    new_Uy = np.concatenate(new_Uy)
    
    return new_Ux, new_Uy, new_inds
        
def create_subsamples(Ux, Uy, ratios_pts, ratios_t, name_file='very_small', size=5509, lstm=False) :
    '''
    Function that subsample the matrix Ux and Uy into different 
    Inputs:
        * Ux (np.array): Matrix of velocities along x at different position on the grid and at different time step (2D matrix)
        * Uy (np.array): Matrix of velocities along y at different position on the grid and at different time step (2D matrix)
        * ratios_pts (list): array of ratio_pts
        * ratios_t (list): array of ratio_t
        * name_file (string): name of the data file (small, very_small, ...)
    Outputs:
        * create a pickle file in the data/pickle folder for each subsampled flattened array
    '''
    for i in range(len(ratios_pts)) :
        for j in range(len(ratios_t)) :
            new_Ux, new_Uy, _ = sample(Ux, Uy, ratios_pts[i], ratios_t[j])
            if lstm:
                flattened_array_sub = flatten_2d(new_Ux, new_Uy, ratios_pts[i])
            else:
                flattened_array_sub = flatten(new_Ux, new_Uy, ratios_pts[i])
            
            name = 'processed_'+str(name_file)+'_'+str(ratios_pts[i])+'_'+str(ratios_t[j])
            cPickle.dump( flattened_array_sub , open( "data/pickle/"+name, "wb" ) )


            
# -----------FLATTENING METHODS------------------

def flatten(Ux, Uy, ratios_pts, size=5509):
    '''
    Function that flattens the original matrices Ux and Uy into a final array of dimension (2*Nu*Nt) * Ns
    Inputs:
        * Ux (np.array): Matrix of velocities along x at different position on the grid and at different time step (2D matrix)
        * Uy (np.array): Matrix of velocities along y at different position on the grid and at different time step (2D matrix)
        * ratio_pts (list): # of positions sampled = ratio* # of positions initially
        * size (int): # of rows (points) belonging to 1 simulation, in our case 1 simulation = 1 datapoint for our neural networks
    Outputs:
        * flattened_array_all (np.array): final flattened array to dimension (2*Nu*Nt) * Ns
    '''
    
    # defining index which separates each simulation with respect to the sampling
    idx_new_sim = np.int(np.floor(ratios_pts * size))
    
    # defining the arrays where we will seperate the different simulation
    simulation_x = []
    simulation_y = []
    
    # for loop to separate the Ux and Uy simulations 
    j = 0
    for i in range(np.int(Ux.shape[0]/idx_new_sim)):
        simulation_x.append([Ux[j:j+idx_new_sim,:]])
        simulation_y.append([Uy[j:j+idx_new_sim,:]])
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


def flatten_2d(Ux, Uy, ratios_pts, size=5509):
    '''
    Function that flattens the original matrices Ux and Uy into a final array of dimension Ns * (2*Nu) * Nt 
    Inputs:
        * Ux (np.array): Matrix of velocities along x at different position on the grid and at different time step (2D matrix)
        * Uy (np.array): Matrix of velocities along y at different position on the grid and at different time step (2D matrix)
        * ratio_pts (list): # of positions sampled = ratio* # of positions initially
        * size (int): # of rows (points) belonging to 1 simulation, in our case 1 simulation = 1 datapoint for our neural networks
    Outputs:
        * flattened_array_all (np.array): final flattened array to dimension  Ns * (2*Nu) * Nt
    '''
    
    # defining index which separates each simulation with respect to the sampling
    idx_new_sim = np.int(np.floor(ratios_pts * size))
    
    # defining the arrays where we will seperate the different simulation
    simulation_x = []
    simulation_y = []
    
    # for loop to separate the Ux and Uy simulations 
    j = 0
    for i in range(np.int(Ux.shape[0]/idx_new_sim)):
        simulation_x.append([Ux[j:j+idx_new_sim,:]])
        simulation_y.append([Uy[j:j+idx_new_sim,:]])
        j = j + idx_new_sim
    simulation_x = np.array(simulation_x).squeeze()
    simulation_y = np.array(simulation_y).squeeze()
    
    # initializing the output flattened array
    flattened_array_all = np.zeros((simulation_x.shape[0],2*simulation_x.shape[1],simulation_x.shape[2]))
    
    # for loop to iterate through the simulations
    for idx_sim in range(simulation_x.shape[0]):
        sim_x = simulation_x[idx_sim]
        sim_y = simulation_y[idx_sim]
        
        # for loop to iterate through the time steps
        for col in range(simulation_x.shape[2]):
            
            # plug the concatenation of the x and y speed coordinate into the column of the final array
            flattened_array_all[idx_sim, :, col] = np.concatenate((sim_x[:,col], sim_y[:,col]), axis=0)
    
    return flattened_array_all
