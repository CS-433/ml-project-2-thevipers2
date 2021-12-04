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
    sum = 0
    for idx, y_val in enumerate(y):
        sum += np.linalg.norm((y_val-y_pred[idx]),2)**2/np.linalg.norm(y_val,2)**2
            
    return sum/ len(y)
    
