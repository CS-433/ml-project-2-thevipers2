# -*- coding: utf-8 -*-
"""cross validation"""
import numpy as np

#Pour l'instant marche pour un seul ratio donné et pour un seul data point

def sample_points(ux, uy, ratios) :
    
    #for 1 data point
    for i, ratio in enumerate(ratios) :
        new_inds = np.random.uniform(0, ux.shape[0], ratio*ux.shape[0])
    
    new_ux = ux[new_inds, :]
    new_uy = uy[new_inds, :]
    
    return ux, uy


def sample_times(ux, uy, ratios) :
        
    #for 1 data point
    for i, ratio in enumerate(ratios) :
        new_inds = np.random.uniform(0, ux.shape[1], ratio*ux.shape[1])
    
    new_ux = ux[:, new_inds]
    new_uy = uy[:, new_inds]
    
    return ux, uy

#get datapoint n from the matrix Ux or Uy :
def get_point(U, n) :
    
    U_n = U[n*5510:(n+1)*5509, :] #A checker si ça marche
    return U_n

def sample(Ux, Uy, ratios_pts, ratios_t) :
    
    new_Ux = []
    new_Uy = []
    for i in range(u.shape[0]/5509) : #A checker
        ux = get_point(Ux, i)
        uy = get_point(Uy, i)
        ux_s1, uy_s1 = sample_points(ux, uy, ratios_pts)
        ux_s2, uy_s2 = sample_times(ux_s1, uy_s1, ratios_t)
        
        new_Ux.append(ux_s2)
        new_Uy.append(uy_s2)  
    new_Ux = np.array(new_Ux)
    new_Uy = np.array(new_Uy)
    
return new_Ux, new_Uy
        
    