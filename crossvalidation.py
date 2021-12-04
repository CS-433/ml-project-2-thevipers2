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
from autoencoder import *

def Kfold(dataset, k_folds, model, epochs, criterion, optimizer) : 
    """
    Perform K-fold cross-validation to estimate the train and test error of the model on the dataset.

    Parameters:
    dataset (np.array): dataset to perform K-fold cross-validation on 
    k_folds (int): number of folds to use for K-fold cross-validation
    model (Pytorch neural network): the Pytorch neural network to cross-validate 
    epochs (int): number of complete cycles through the entire dataset the machine learning algorithm complete during training
    criterion (method from nn.Module to estimate the loss): loss to use during training 
    optimizer (optimizer from torch.optim): optimization algorithm to use during training 

    Returns:
    test_error (): the average test error obtained during K-fold cross-validation 

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

    Parameters:
    dataset (np.array): dataset to perform K-fold cross-validation on 
    k_folds (int): number of folds to use for K-fold cross-validation
    model (Pytorch neural network): the Pytorch neural network to cross-validate
    epochs (int): number of complete cycles through the entire dataset the machine learning algorithm complete during training
    criterion (method from nn.Module to estimate the loss): loss to use during training 
    optimizer (optimizer from torch.optim): optimization algorithm to use during training 
    number_neurons (np.array): the different number of neurons in the latent layer we want to test 

    Returns:
    results (): the average test error obtained for each number of neurons 
    best_result (): the best test error obtained
    best_neuron_number (): the number of neuron in the latent layer that leads to the best test error

    """
    results = []
    for neuron in number_neurons : 
        res = Kfold_2(dataset, k_folds, model, epochs, criterion, optimizer)
        results.append(res)
    best_result = np.min(results)
    best_neuron_number = np.argmin(results)
    return results, best_result, best_neuron_number  