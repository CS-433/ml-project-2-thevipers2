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
    def __init__(self, epochs=100, batchSize=128, learningRate=1e-3) :
        super(Autoencoder, self).__init__()
        # initializing network parameters
        self.epochs = epochs                               
        self.batchSize = batchSize
        self.learningRate = learningRate
        
        # encoder network architecture with 4 linear layers
        self.encoder = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(True),
        nn.Linear(128, 64),
        nn.ReLU(True),
        nn.Linear(64, 12),
        nn.ReLU(True),
        nn.Linear(12, 3)
        )
        
        # decoder network architecture with 4 linear layers
        self.decoder = nn.Sequential(
        nn.Linear(3, 12),
        nn.ReLU(True),
        nn.Linear(12, 64),
        nn.ReLU(True),
        nn.Linear(64, 128),
        nn.ReLU(True),
        nn.Linear(128, 784),
        nn.ReLU(True) # que mettre en output layer ? ReLU ? Tanh ? Sigmoid ? 
        )
        
        # optimizer and loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        
    def forward(self,x) : 
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def train() : 
         for epoch in range(self.epochs) : 
                # récupérer les inputs 
                '''
                '''
                # predictions
                output = self(data)
                # calculate loss
                loss = self.criterion(output, data)
                # backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self.epochs, loss.data))