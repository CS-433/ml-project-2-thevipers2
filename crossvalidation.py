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
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from autoencoder import *
import math
from os import linesep as endl

def Kfold(dataset, k_folds, input_size, epochs, criterion, learningRate, neuron=5, momentum=0.9, comment = True):
    """
    Perform K-fold cross-validation to estimate the train and test error of the model on the dataset.

    Inputs:
        * dataset (np.array): dataset to perform K-fold cross-validation on 
        * k_folds (int): number of folds to use for K-fold cross-validation
        * input_size (int): the size of the input  
        * epochs (int): number of complete cycles through the entire dataset the neural network completes during training
        * criterion (method from nn.Module to estimate the loss): loss to use during training 
        * learningRate (float): learning rate 
        * neuron (int): number of neurons in the latent layer
        * momentum (float): momentum for the optimizer 
        * comment (boolean): True to show graphs of training and test error in function of epochs number

    Outputs:
        * mean_test_err (float): the average test error obtained during K-fold cross-validation 
    """
    
    torch.manual_seed(0)
    
    # define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state = 1)
    foldperf={}
    f_test, f_train = [], []

    # iterate through the folds
    for fold, (train_idx,val_idx) in enumerate(kfold.split(np.arange(len(dataset)))):

        # define the model
        model = Autoencoder(input_size, neuron)
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-5)
        # print
        if(comment) :
            print('--------------------------------')
            print(f'FOLD {fold}')
            print('--------------------------------')
        
        # sample the elements from train_idx and from val_idx and then we convert these samplers into DataLoader objects
        train_loader = DataLoader(dataset, batch_size=20, sampler=train_idx)
        test_loader = DataLoader(dataset, batch_size=20, sampler=val_idx)
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
            
            if(comment) :
                print("Epoch:{}/{} Training Error:{:.3f} Test Error:{:.3f}".format(epoch + 1,epochs,train_error,test_error))
            history['train_error'].append(train_error)
            history['test_error'].append(test_error)

        f_test.append(test_error)
        f_train.append(train_error)
        foldperf['fold{}'.format(fold+1)] = history  

    # compute the average relative errors over all the folds 
    testl_f,tl_f=[],[]
    for f in range(1,k_folds+1):
        tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_error']))
        testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_error']))
    
    print('Performance of {} fold cross validation: '.format(k_folds))
    print("Average Training Error: {:.3f} \t Average Test Error: {:.3f}".format(np.mean(f_train),np.mean(f_test)))
    print(' ')
    
    diz_ep = {'train_error_ep':[],'test_error_ep':[]}

    for i in range(epochs):
          diz_ep['train_error_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_error'][i] for f in range(k_folds)]))
          diz_ep['test_error_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['test_error'][i] for f in range(k_folds)]))
    
    if(comment) :
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
    
    mean_test_err = np.mean(f_test)
    
    return mean_test_err  



def tuning_latent_layer(dataset, k_folds, input_size, epochs, criterion, lr, number_neurons, dataset_name_="very_small", plot=True) : 
    """
    Perform K-fold cross-validation to find the best number of neurons in the latent space.

    Inputs:
        * dataset (np.array): dataset to perform K-fold cross-validation on 
        * k_folds (int): number of folds to use for K-fold cross-validation
        * epochs (int): number of complete cycles through the entire dataset the neural network completes during training
        * criterion (method from nn.Module to estimate the loss): loss to use during training 
        * learningRate (float): learning rate 
        * number_neurons (np.array): the different number of neurons in the latent layer we want to test 
        * dataset_name (string): dataset name used to save the final plot 
        * plot (boolean): True to show graphs of test error in function of the number of neurons
    OUTPUTS:
        * best_result (float): the best test error obtained
        * best_neuron_number (int) : number of neurons in the latent space that gives the best test error
    """
    results = []
    for neuron in number_neurons :
        print('\033[1m'+'Number of neurons = ', str(neuron))
        print('\033[0m')
        res = Kfold(dataset, k_folds, input_size, epochs, criterion, lr, neuron, comment = False)
        results.append(res)
    best_result = np.min(results)
    best_neuron_number = number_neurons[np.argmin(results)]
    
    print(f"The results obtained for the number of latent neurons tested are the following : {results}.")
    print(f"The best average test error obtained is {best_result}, and it is obtained with {best_neuron_number} neurons in the latent layer.")
    
    if(plot) :
        plt.plot(number_neurons, results, 'bo')
        new_list = range(math.floor(min(number_neurons)), math.ceil(max(number_neurons))+1)
        plt.xticks(new_list)
        plt.plot(best_neuron_number, best_result, 'ro', markersize=8, label = 'Best number of neurons : '+str(best_neuron_number)+endl +'error : ' + str(round(best_result, 3)))
        plt.xlabel('Number of neurons') ; plt.ylabel('Test error')
        title = 'Average test error on the ' + str(k_folds) + '-fold for different number of neurons'
        plt.title(title)
        plt.legend()
        plt.savefig("Latent_neurons_tuning_"+dataset_name_)
        plt.show()
    return best_result, best_neuron_number



def tuning_lr(dataset, k_folds, input_size, epochs, criterion, learning_rates, dataset_name_="very_small", plot = True) :
    """
    Perform K-fold cross-validation to find the best learning rate of our model.

    Inputs:
        * dataset (np.array): dataset to perform K-fold cross-validation on 
        * k_folds (int): number of folds to use for K-fold cross-validation
        * epochs (int): number of complete cycles through the entire dataset the neural network completes during training
        * criterion (method from nn.Module to estimate the loss): loss to use during training 
        * learning_rates (list): learning rates to tune for the optimizer
        * momentum (float): momentum for the SGD optimizer 
        * dataset_name (string): dataset name used to save the final plot 
        * plot (boolean): True to show graphs of test error in function of the learning rate (in log space)
    Outputs:
        * best_result (float): the best test error obtained
        * best_learning_rate (float) : learning rate that gives the best test error
    """
    
    best_result = 100000
    best_learning_rate = 0
    results = []
    for learning_rate in learning_rates :
        print('\033[1m'+'Learning rate = ', str(learning_rate))
        print('\033[0m')
        res = Kfold(dataset, k_folds, input_size, epochs, criterion, learning_rate, comment = False)
        results.append(res)
        if(res < best_result) :
            best_result = res
            best_learning_rate = learning_rate
    print('\033[0m')
    print('Best learning rate is ', best_learning_rate, ' with a best error of : ', best_result)
    print('\033[0m')
    
    if(plot) :
        plt.plot(learning_rates, results, 'bo')
        plt.plot(best_learning_rate, best_result, 'ro', markersize=8, label = 'Best learning rate :'+str(best_learning_rate)+ endl +'error : ' + str(round(best_result, 3)))
        plt.xscale('log')
        plt.xlabel('Learning rate') ; plt.ylabel('Test error')
        title = 'Average test error on the ' + str(k_folds) + '-fold for different learning rates'
        plt.title(title)
        plt.legend()
        plt.savefig("Learning_rate_tuning_"+dataset_name_)
        plt.show()
    return best_result, best_learning_rate


def tuning_lr_momentum(dataset, k_folds, input_size, epochs, criterion, learning_rates, momentums) :
    """
    Perform K-fold cross-validation to .

    Inputs:
        * dataset (np.array): dataset to perform K-fold cross-validation on 
        * k_folds (int): number of folds to use for K-fold cross-validation
        * model (Pytorch neural network): the Pytorch neural network to cross-validate
        * epochs (int): number of complete cycles through the entire dataset the neural network completes during training
        * criterion (method from nn.Module to estimate the loss): loss to use during training 
        * optimizer (optimizer from torch.optim): optimization algorithm to use during training 
        * learning_rates (list): learning rates to tune for the optimizer
        * momentums (list): momentums to tune for the SGD optimizer 

    Outputs:
        * results (np.array): the average test error obtained for each number of neurons 
        * best_result (float): the best test error obtained
        * best_learning_rate (int): best learning rate found after cross-validation
        * best_momentum (int): best momentum found after cross-validation
    """
    
    best_result = 100000
    best_learning_rate = 0
    best_momentum = 0
    for momentum in momentums : 
        print('\033[1m'+'Momentum = ', str(momentum))
        print('\033[0m')
        for learning_rate in learning_rates :
            print('\033[1m'+'Learning rate = ', '\033[1m'+str(learning_rate))
            res = Kfold(dataset, k_folds, input_size, epochs, criterion, learning_rate, momentum, comment = False)
            if(res < best_result) :
                best_result = res
                best_learning_rate = learning_rate
                best_momentum = momentum
    print('Best learning rate is ', best_learning_rate,' with a best momentum of ', best_momentum, ' with a best error of : ', best_result)
    return best_result, best_learning_rate, best_momentum


