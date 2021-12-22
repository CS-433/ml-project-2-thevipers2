# -*- coding: utf-8 -*-
"""
Autoencoder code.
"""
import numpy as np 
import matplotlib.pyplot as plt
import torch                                            
from torch import nn                       
from torch.autograd import Variable 


class Autoencoder(nn.Module) :
    def __init__(self, input_size, neuron=5) :
        """
        Model initialisation.
        Inputs:
            * input_size : the number of expected features in the input x, corresponds to time-steps of the input matrices
            * neuron : the number of neurons in our latent layer
        """
        super().__init__()
        
        # encoder network architecture
        self.encoder = nn.Sequential(
        nn.Linear(input_size, 256),
        nn.Linear(256, 128), 
        nn.Linear(128, 64),
        nn.Linear(64, 32),
        nn.Linear(32, 16), 
        nn.Linear(16, neuron)
        )
        
        # decoder network architecture
        self.decoder = nn.Sequential(
        nn.Linear(neuron, 16),
        nn.Linear(16, 32),
        nn.Linear(32, 64),
        nn.Linear(64, 128),
        nn.Linear(128, 256),
        nn.Linear(256, input_size)
        )
        
    def forward(self, x, show_compression=False):
        """
        Run forward computation.
        Inputs:
            * x (torch.Tensor): tensor of input data
            * show_compression (boolean): true to return x 
        Outputs: 
            * x (torch.Tensor): final output
        """
        x = self.encoder(x)
        if(show_compression) :
            return x
        x = self.decoder(x)
        return x


def train_epoch(input_data, net, criterion, optimizer) :
    """
    Train the neural network.

    Inputs:
        * input_data (np.array): dataset to train the neural network 
        * net (Pytorch neural network): the neural network to train
        * criterion (method from nn.Module to estimate the loss): loss to use during training 
        * optimizer (optimizer from torch.optim): optimization algorithm to use during training 
    Outputs:
        * train_loss(float): final loss 
    """
    
    # initialize the parameters
    train_loss= 0.0
    net.train()
    
    for sim in input_data : # loop over the data points (simulations) in the dataset 
        # predictions
        sim = sim.float()
        output = net(sim)
        # calculate loss
        loss = criterion(output, sim)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update loss
        train_loss +=loss.item()
    
    return train_loss

    
def valid_epoch(test_data, net):
    """
    Evaluate the neural network.

    Inputs:
        * test_data (np.array): dataset to evaluate using the trained neural network 
        * net (Pytorch neural network): the neural network to evaluate
        
    Outputs:
        * err (float): the relative test error 
    """
    
    # initialize the parameters
    net.eval()        
    pred = []
    
    with torch.no_grad():
        for data in test_data :
            data = data.float()
            # predict the output 
            predicted = net(data)
            pred.append(predicted)
        # compute the relative error    
        err = (relative_error(test_data, pred))

    return err  


def relative_error(y, y_pred) : 
    """
    Evaluate the relative error.

    Inputs:
        * y (np.array): the true outputs (equal to the inputs in the autoencoder) 
        * y_pred (np.array): the predicted outputs
        
    Outputs:
        * rel_err (float): the relative test error 
    """
    
    sum = 0
    for idx, y_val in enumerate(y):
        sum += np.linalg.norm((y_val-y_pred[idx]),2)**2/np.linalg.norm(y_val,2)**2
            
    rel_err = sum/ len(y)
    return rel_err
    
