# -*- coding: utf-8 -*-
"""
Autoencoder code.
"""
import numpy as np 
import matplotlib.pyplot as plt
import torch                                            
from torch import nn                       
from torch.autograd import Variable 
      
    
# Encoder Class
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Encoder initialisation.
        Inputs:
            * input_size (int): the number of expected features in the input x, corresponds to time-steps of the input matrices
            * hidden_size (int): the number of features in the hidden state h, also translate to the number of neurons on our latent layer in our case 
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm_encoder_1 = nn.LSTM(input_size=input_size, hidden_size=48, batch_first=True)
        self.lstm_encoder_2 = nn.LSTM(input_size=48, hidden_size=32, batch_first=True)
        self.lstm_encoder_3 = nn.LSTM(input_size=32, hidden_size=hidden_size, batch_first=True)
        

    def forward(self, x):
        """
        Run forward computation.
        Inputs:
            * x (torch.Tensor): tensor of input data
        Outputs: 
            * x_enc (torch.Tensor): final hidden state 
            * out (torch.Tensor): final output
        """
        x, (_, _) = self.lstm_encoder_1(x)
        x, (_, _) = self.lstm_encoder_2(x)
        out, (last_h_state, last_c_state) = self.lstm_encoder_3(x)
        x_enc = last_h_state.squeeze(dim=0)
        x_enc = x_enc.unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_enc, out
        


# Decoder Class
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Decoder initialisation.
        Inputs:
            * input_size : the number of expected features in the input x, corresponds to time-steps of the input matrices
            * hidden_size : the number of features in the hidden state h, also translate to the number of neurons on our latent layer in our case 
        """
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm_decoder_1 = nn.LSTM(input_size=hidden_size, hidden_size=32, batch_first=True)
        self.lstm_decoder_2 = nn.LSTM(input_size=32, hidden_size=48, batch_first=True)
        self.lstm_decoder_3 = nn.LSTM(input_size=48, hidden_size=input_size, batch_first=True)
        

    def forward(self, z):
        """
        Run forward computation.
        Inputs:
            * z (torch.Tensor): tensor of input data
        Outputs: 
            * hidden_state (torch.Tensor): final hidden state 
            * dec_out (torch.Tensor): final output
        """
        # z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        z, (_, _) = self.lstm_decoder_1(z)
        z, (_, _) = self.lstm_decoder_2(z)
        dec_out, (hidden_state, cell_state) = self.lstm_decoder_3(z)
        return dec_out, hidden_state


# LSTM Auto-Encoder Class
class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Model initialisation.
        Inputs:
            * input_size : the number of expected features in the input x, corresponds to time-steps of the input matrices
            * hidden_size : the number of features in the hidden state h, also translate to the number of neurons on our latent layer in our case 
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size)
        self.decoder = Decoder(input_size=input_size, hidden_size=hidden_size)

    def forward(self, x, return_last_h=False, return_enc_out=False):
        """
        Run forward computation.
        Inputs:
            * x (torch.Tensor): tensor of input data
            * return_last_h (boolean): true to return final hidden state
            * return_enc_out (boolean): true to return final cell state 
        Outputs: 
            * x_dec (torch.Tensor): final output
        """
        x_enc, enc_out = self.encoder(x)
        x_dec, last_h = self.decoder(x_enc)

        if return_last_h:
            return x_dec, last_h
        elif return_enc_out:
            return x_dec, enc_out
        return x_dec 
    
    
def train_epoch_lstm(input_data, net, criterion, optimizer) :
    """
    Train the LSTM neural network.

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

    
def valid_epoch_lstm(test_data, net):
    """
    Evaluate the LSTM neural network.

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
        err = (relative_error_lstm(test_data, pred))

    return err  


def relative_error_lstm(y, y_pred) : 
    """
    Evaluate the relative error for the LSTM model.

    Inputs:
        * y (np.array): the true outputs (equal to the inputs in the autoencoder) 
        * y_pred (np.array): the predicted outputs
        
    Outputs:
        * rel_err (float): the relative test error 
    """
    
    sum = 0
    i = 0
    for idx, y_val in enumerate(y):
        for idx_2, y_one in enumerate(y_val):
            sum += np.linalg.norm((y_one-y_pred[idx][idx_2]), 2)**2/(np.linalg.norm(y_one,2)**2)
            i += 1
            
    rel_err = sum / (i)
    return rel_err
    

                              
                                
                                
                                
                                