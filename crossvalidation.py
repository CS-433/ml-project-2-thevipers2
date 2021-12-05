# -*- coding: utf-8 -*-
"""
Cross-validation code.
"""
import numpy as np 
import matplotlib.pyplot as plt
import torch                                            
from torch import nn                       
from torch.autograd import Variable 
from sklearn.model_selection import KFold
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from autoencoder import *

def Kfold(dataset, k_folds, model, epochs, criterion, optimizer) : 
    """
    Perform K-fold cross-validation to estimate the train and test error of the model on the dataset.

    Inputs:
        * dataset (np.array): dataset to perform K-fold cross-validation on 
        * k_folds (int): number of folds to use for K-fold cross-validation
        * model (Pytorch neural network): the Pytorch neural network to cross-validate 
        * epochs (int): number of complete cycles through the entire dataset the neural network complete during training
        * criterion (method from nn.Module to estimate the loss): loss to use during training 
        * optimizer (optimizer from torch.optim): optimization algorithm to use during training 

    Outputs:
        * test_error (float): the average test error obtained during K-fold cross-validation 

    """

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    foldperf={}

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        history = {'train_error': [], 'test_error': []}
        train(dataset[train_ids], model, epochs, criterion, optimizer)
        train_loss=test(dataset[train_ids], model, epochs, criterion, optimizer)
        test_loss=test(dataset[test_ids], model, epochs, criterion, optimizer)


        history['train_error'].append(train_loss)
        history['test_error'].append(test_loss)
        print(history)

        foldperf['fold{}'.format(fold+1)] = history 
        
        
    testl_f,tl_f=[],[]

    for f in range(1,k_folds+1):
        tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_error']))
        testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_error']))


    print('Performance of {} fold cross validation: '.format(k_folds))
    print("Average Training Error: {:.3f} \t Average Test Error: {:.3f}".format(np.mean(tl_f),np.mean(testl_f)))
    
    return np.mean(testl_f)
       
        

def Kfold_latent_layer(dataset, k_folds, model, epochs, criterion, optimizer, number_neurons) : 
    """
    Perform K-fold cross-validation to .

    Inputs:
        * dataset (np.array): dataset to perform K-fold cross-validation on 
        * k_folds (int): number of folds to use for K-fold cross-validation
        * model (Pytorch neural network): the Pytorch neural network to cross-validate
        * epochs (int): number of complete cycles through the entire dataset the machine learning algorithm complete during training
        * criterion (method from nn.Module to estimate the loss): loss to use during training 
        * optimizer (optimizer from torch.optim): optimization algorithm to use during training 
        * number_neurons (np.array): the different number of neurons in the latent layer we want to test 

    Outputs:
        * results (np.array): the average test error obtained for each number of neurons 
        * best_result (float): the best test error obtained
        * best_neuron_number (int): the number of neuron in the latent layer that leads to the best test error
    """
    results = []
    for neuron in number_neurons : 
        res = Kfold_2(dataset, k_folds, model, epochs, criterion, optimizer)
        results.append(res)
    best_result = np.min(results)
    best_neuron_number = np.argmin(results)
    return results, best_result, best_neuron_number  

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
    



def tuning(config):
    # Data Setup
    
    train_loader = DataLoader(
        datasets(
        "flattened_array_train.npy",
            loader=np.load, 
        transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=18)
    test_loader = DataLoader(
        datasets(
        "flattened_array_test.npy",
            loader=np.load,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=18)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # à modulariser avec global? 
    input_data = 121000
    criterion = nn.MSELoss() 
    model = Autoencoder(input_data)
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"])
    
    for i in range(10):
        train(input_data, net=model, epochs=config['epochs'], criterion=criterion,optimizer=optimizer)
        acc = test(model, test_loader)

        # Send the current training result back to Tune
        tune.report(mean_accuracy=acc)

        if i % 5 == 0:
            # This saves the model to the trial directory
            torch.save(model.state_dict(), "./model.pth")
            

