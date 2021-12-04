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
    def __init__(self, input_size) :
        super().__init__()
        
        # encoder network architecture with 4 linear layers
        self.encoder = nn.Sequential(
        nn.Linear(input_size, 12),
        nn.ReLU(True),
        nn.Linear(12, 5),
        )
        
        # decoder network architecture with 4 linear layers
        self.decoder = nn.Sequential(
        nn.Linear(5, 12),
        nn.ReLU(True),
        nn.Linear(12, input_size),
        nn.ReLU(True)
        )
        
    def forward(self,x) : 
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(input_data, net, epochs, criterion, optimizer) :
    """
    Train the neural network.

    Inputs:
        * input_data (np.array): dataset to train the neural network 
        * net (Pytorch neural network): the neural network to train
        * epochs (int): number of complete cycles through the entire dataset the neural network completes during training
        * criterion (method from nn.Module to estimate the loss): loss to use during training 
        * optimizer (optimizer from torch.optim): optimization algorithm to use during training 

    """
    losses = []
    for epoch in range(epochs) : # loop over the dataset multiple times
        # recover the inputs 
        data = torch.from_numpy(input_data)
        for sim in data : # loop over the data points (simulations) in the dataset 
            # predictions
            sim = sim.float()
            output = net(sim)
            # calculate loss
            loss = criterion(output, sim)
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))  
        # Storing the losses in a list for plotting
        losses.append(loss.detach().numpy())
        
    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    # Plotting the loss decay
    plt.plot(losses)
    
    
def test(test_data, net, epochs, criterion, optimizer) : 
    """
    Evaluate the neural network.

    Inputs:
        * test_data (np.array): dataset to evaluate using the trained neural network 
        * net (Pytorch neural network): the neural network to evaluate
        * epochs (int): number of complete cycles through the entire dataset the neural network completes during training
        * criterion (method from nn.Module to estimate the loss): loss to use during training 
        * optimizer (optimizer from torch.optim): optimization algorithm to use during training 
        
    Outputs:
        * err (int): the relative test error 

    """
    pred = []
    with torch.no_grad():
        test_data = torch.from_numpy(test_data)
        for data in test_data :
            data = data.float()
            predicted = net(data)
            pred.append(predicted)
        err = (relative_error(test_data, pred))

    return err
    


def relative_error(y, y_pred) : 
    """
    Evaluate the relative error.

    Inputs:
        * y (np.array): the true outputs (equal to the inputs in the autoencoder) 
        * y_pred (np.array): the predicted outputs
        
    Outputs:
        * rel_err (int): the relative test error 

    """
    sum = 0
    for idx, y_val in enumerate(y):
        sum += np.linalg.norm((y_val-y_pred[idx]),2)**2/np.linalg.norm(y_val,2)**2
            
    rel_err = sum/ len(y)
    return rel_err
    
