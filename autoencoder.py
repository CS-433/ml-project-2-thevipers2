# -*- coding: utf-8 -*-
"""
Autoencoder code.
"""
import numpy as np 
import matplotlib.pyplot as plt
import torch                                            
from torch import nn                       
from torch.autograd import Variable 

# define network parameters
learningRate=1e-1
epochs=100

class Autoencoder(nn.Module) :
    def __init__(self) :
        super().__init__()
        
        # encoder network architecture with 4 linear layers
        self.encoder = nn.Sequential(
        nn.Linear(121000, 12),
        nn.ReLU(True),
        nn.Linear(12, 5),
        )
        
        # decoder network architecture with 4 linear layers
        self.decoder = nn.Sequential(
        nn.Linear(5, 12),
        nn.ReLU(True),
        nn.Linear(12, 121000),
        nn.ReLU(True)
        )
        
    def forward(self,x) : 
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

net = Autoencoder()
optimizer = torch.optim.Adam(net.parameters(), lr=learningRate, weight_decay=1e-5)
criterion = nn.MSELoss()  

def train(input_data) : 
     for epoch in range(epochs) : # loop over the dataset multiple times
        # récupérer les inputs 
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
