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
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from autoencoder import *


def Kfold(dataset, k_folds, input_size, epochs, criterion, learningRate):
    """
    Perform K-fold cross-validation to estimate the train and test error of the model on the dataset.

    Inputs:
        * dataset (np.array): dataset to perform K-fold cross-validation on 
        * k_folds (int): number of folds to use for K-fold cross-validation
        * input_size (int): the size of the input  
        * epochs (int): number of complete cycles through the entire dataset the neural network completes during training
        * criterion (method from nn.Module to estimate the loss): loss to use during training 
        * learningRate (float): learning rate  

    Outputs:
        * mean_test_err (float): the average test error obtained during K-fold cross-validation 
    """
    
    # define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state = 1)
    foldperf={}
   
    # iterate through the folds
    for fold, (train_idx,val_idx) in enumerate(kfold.split(np.arange(len(dataset)))):

        # define the model
        model = Autoencoder(input_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-5)
        # print
        print('--------------------------------')
        print(f'FOLD {fold}')
        print('--------------------------------')
        # sample the elements from train_idx and from val_idx and then we convert these samplers into DataLoader objects
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=10, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=10, sampler=test_sampler)
        # initialize the dictionary and the array to store the errors
        history = {'train_error': [], 'test_error': []}

        # loop over the epochs
        for epoch in range(epochs):
            # train the model 
            train_loss=train_epoch(train_loader,model, criterion,optimizer)
            # compute the relative training error
            train_error = valid_epoch(train_loader,model)
            # compute the relative test error
            test_error=valid_epoch(test_loader,model)

            print("Epoch:{}/{} Training Error:{:.3f} Test Error:{:.3f}".format(epoch + 1,epochs,train_error,test_error))
            history['train_error'].append(train_error)
            history['test_error'].append(test_error)

        foldperf['fold{}'.format(fold+1)] = history  

    # compute the average relative errors over all the folds 
    testl_f,tl_f=[],[]
    for f in range(1,k_folds+1):
        tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_error']))
        testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_error']))

    print('Performance of {} fold cross validation: '.format(k_folds))
    print("Average Training Error: {:.3f} \t Average Test Error: {:.3f}".format(np.mean(tl_f),np.mean(testl_f)))
    
    diz_ep = {'train_error_ep':[],'test_error_ep':[]}

    for i in range(epochs):
          diz_ep['train_error_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_error'][i] for f in range(k_folds)]))
          diz_ep['test_error_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['test_error'][i] for f in range(k_folds)]))

    # Plot training and test relative errors
    plt.figure(figsize=(10,8))
    plt.semilogy(diz_ep['train_error_ep'], label='Train')
    plt.semilogy(diz_ep['test_error_ep'], label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    #plt.grid()
    plt.legend()
    plt.title('Autoencoder error')
    plt.show()
    
    mean_test_err = np.mean(testl_f)
    
    return mean_test_err        



def Kfold_latent_layer(dataset, k_folds, model, epochs, criterion, optimizer, number_neurons) : 
    """
    Perform K-fold cross-validation to .

    Inputs:
        * dataset (np.array): dataset to perform K-fold cross-validation on 
        * k_folds (int): number of folds to use for K-fold cross-validation
        * model (Pytorch neural network): the Pytorch neural network to cross-validate
        * epochs (int): number of complete cycles through the entire dataset the neural network completes during training
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
        res = Kfold(dataset, k_folds, model, epochs, criterion, optimizer)
        results.append(res)
    best_result = np.min(results)
    best_neuron_number = np.argmin(results)
    return results, best_result, best_neuron_number 



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
            

