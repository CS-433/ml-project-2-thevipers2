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
learningRate=1e-3
epochs=100

class Autoencoder(nn.Module) :
    def __init__(self, inputsize) :
        super().__init__()
        
        # encoder network architecture with 4 linear layers
        self.encoder = nn.Sequential(
        nn.Linear(inputsize, 25),
        nn.ReLU(True),
        nn.Linear(25, 12),
        nn.ReLU(True),
        nn.Linear(12, 8),
        nn.ReLU(True),
        nn.Linear(8, 5)
        )
        
        # decoder network architecture with 4 linear layers
        self.decoder = nn.Sequential(
        nn.Linear(5, 8),
        nn.ReLU(True),
        nn.Linear(8, 12),
        nn.ReLU(True),
        nn.Linear(12, 25),
        nn.ReLU(True),
        nn.Linear(25, inputsize),
        nn.ReLU(True)
        )
        
    def forward(self,x) : 
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def train(input_data) : 
     for epoch in range(epochs) : # loop over the dataset multiple times
        # récupérer les inputs 
        data = torch.from_numpy(input_data)
        # optimizer and loss
        net = Autoencoder(inputsize = (input_data).shape[1])
        optimizer = torch.optim.Adam(net.parameters(), lr=learningRate, weight_decay=1e-5)
        criterion = nn.MSELoss()  
        # predictions
        data = data.float()
        output = net(data)
        # calculate loss
        loss = criterion(output, data)
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))           
