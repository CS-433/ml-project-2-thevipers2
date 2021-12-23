# -*- coding: utf-8 -*-
"""
Helpers functions
"""
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
import _pickle as cPickle
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from autoencoder import *
from crossvalidation import *
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import os 

warnings.filterwarnings('ignore')


def load_csv_data(data_path):
    """
    Loads data and returns X (which is also Y in our case) as a numpy array
    Inputs:
        * data_path (string): path to data file
    Output: 
        * x (np.array): numpy data array
    """
    x = np.genfromtxt(data_path, delimiter=",")
    return x

def get_Nu_Nt_sampled(U, U_sampled) :
    """
    Get the number of rows in the new U matrix after sampling
    Inputs :
        * U (np.array): Original matrix U
            # of rows : nb_of_datapoints * nb_rows_of_1_datapoint_before_sampling = nb_of_datapoints * 5509 
            # of columns : nb_of_time_steps_before_sampling
        * U_sampled : Sampled version of the matrix U
            # of rows : nb_of_datapoints * nb_rows_of_1_datapoint_after_sampling =  nb_of_datapoints * 5509 * ratio_pts
            # of columns : nb_of_time_steps_after_sampling = nb_of_time_steps_before_sampling * ratio_times
            (Recall : ratio_times/ratio_pts = ratio of times/positions that we keep in U_sampled)
    Outputs :
        * Nu_sampled : nb_rows_of_1_datapoint_after_sampling = 5509 * ratio_pts
    """

    nb_of_pts = U.shape[0]/5509
    
    return int(U_sampled.shape[0]/nb_of_pts), int(U_sampled.shape[1])


def plot_sampled_coord(indices, coord_path='data/coordinates.csv') :
    '''
    Plot the sampled positions on a graph with coordinates x in function of coordinates y
    Inputs:
        * indices (list): indices of points to draw
        * coord_path (string): path to coordinates of mesh data points
    '''
    
    coords = np.genfromtxt(coord_path, delimiter=",")
    np.nan_to_num(coords, False)
    
    new_x = coords[indices, 0]
    new_y = coords[indices, 1]
    
    plt.figure(figsize=(9,5)) 
    sns.scatterplot(new_x, new_y, color = 'red', alpha = 1, label='Sampled points')
    sns.scatterplot(coords[:,0], coords[:,1], color = 'k', alpha = .5, s=5, label='Points of the meshgrid')
    plt.xlabel('x', fontsize=20) ; plt.ylabel('y', fontsize=20)
    title = 'Sampled positions of the simulation (' + str(round((len(indices)/5509)*100, 1)) + '% of the positions)'
    plt.title(title, fontsize=16)
    plt.legend(fontsize=15)
    plt.savefig("Distribution of the sampled positions")
    plt.show()

    
def physical_params_pipeline(dataset_name= 'data/pickle/middle_small/processed_middle_small_0.1_0.25', params_name= "Data/params_middle_small.csv.bz2", epochs=30, degree=3, alpha_=1,  learning_rate = 1e-4, neuron_=10, criterion = nn.MSELoss(), seed=1) :

    '''
    This pipeline tries to fit a regression model to the compressed data (solution of the encoder) with the true parameters of 
    the haemodynamic problem at each each training epoch of the auto-encoder
    
    This pipeline is meant to do the following steps :
        - Load the subdataset we want to use and load the real parameters that generated this dataset (solutions of the 
            haemodynamic problem)
        - Split the dataset in x_train and x_test, same for the parameters, they will the "y" of our future regression problem
        - Train/Test the autoencoder for different epoch
        - For each epoch we fit a regression and a regularized regression with X = solution of the encoder (compression) and 
            Y = real parameters of the problem
        - We can add polynomial extensions to our fit
    Inputs:
        * dataset_name : subdataset we want to use
        * params_name : real parameters that generated this dataset
        * epochs : maximum number of epochs
        * degree : degree of polynomial extension
        * alpha_ : ridge regularization factor
        * learning_rate : learning rate for the auto-encoder
        * neuron_ : number of neurons in the latent space
        * criterion : metric used for the optimization of the auto-encoder  (default MSE loss)
        * seed : fix the train/test split
    Outputs :
        * test_scores : R^2 of simple regression model on the test set
        * test_scores_ridge : R^2 of ridge regression model on the test set
        * train_scores : R^2 of simple regression model on the test set
        * train_scores_ridge : R^2 of ridge regression model on the train set
    '''
    
    #Load the real parameters of the haemodynamic problem
    compressed_Y_pd = pd.read_csv(params_name, header=None)
    comp_Y = compressed_Y_pd.to_numpy()
    comp_Y_train, comp_Y_test = train_test_split(comp_Y, test_size=0.1, random_state=seed, shuffle=False)

    #Load the dataset (solutions of the haemodynamic problem)
    x  = cPickle.load(open(dataset_name, "rb"))

    #Split
    x_train, x_test = train_test_split(x, test_size=0.1, random_state=seed, shuffle=False)
    y_train, y_test = x_train, x_test

    #Declare the model
    input_size=x_train.shape[1]
    model = Autoencoder(input_size, neuron_)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    #Dataloader
    train_loader = DataLoader(x_train, batch_size=20)
    test_loader = DataLoader(x_test, batch_size=20)

    #Training and testing over the epochs
    test_scores = []
    test_scores_ridge = []
    train_scores = []
    train_scores_ridge = []
    test_scores_elasticNet = []
    train_scores_elasticNet = []
    for epoch in range(epochs):
        # train the model 
        train_loss=train_epoch(train_loader, model, criterion, optimizer)
        # compute the relative training error
        train_error = valid_epoch(train_loader, model)
        # compute the relative test error
        test_error=valid_epoch(test_loader, model)

        print('\033[1m')
        print("Epoch:{}/{} Training Error:{:.3f} Test Error:{:.3f}".format(epoch + 1,epochs,train_error,test_error))
        print('\033[0m')

        #Extract the compression
        compressed_X_train = []
        for train in train_loader :
            train = train.float()
            compressed_x_train = model.forward(x=train, show_compression=True)
            compressed_X_train.append(compressed_x_train.detach().numpy())

        compressed_X_test = []    
        for test in test_loader :
            test = test.float()
            compressed_x_test = model.forward(x=test, show_compression=True)
            compressed_X_test.append(compressed_x_test.detach().numpy())       

        #Concatenate testing and training set
        comp_X_train = np.concatenate(np.squeeze(np.array(compressed_X_train)))
        comp_X_test = (np.squeeze(np.array(compressed_X_test)))

        #Polynomial extension
        exponents = np.arange(degree) + 1

        comp_X_train_pol = []
        for row in comp_X_train: 
            comp_X_train_pol.append((row**exponents[:,None]).ravel())

        comp_X_test_pol = []
        for row in comp_X_test: 
            comp_X_test_pol.append((row**exponents[:,None]).ravel())  

        #Regression models
        print('\033[1m'+'For the simple regression model : '+'\033[0m')
        reg_pol = LinearRegression().fit(comp_X_train, comp_Y_train)
        train_score = (reg_pol.score(comp_X_train, comp_Y_train))
        test_score = (reg_pol.score(comp_X_test, comp_Y_test))
        test_scores.append(test_score)
        train_scores.append(train_score)
        print('The train score is: ', train_score)
        print('The test score is: ', test_score)

        print('\033[1m'+'For the Ridge regression model : '+'\033[0m')
        ridge_pol = Ridge(alpha=alpha_).fit(comp_X_train_pol, comp_Y_train)
        train_score_ridge = (ridge_pol.score(comp_X_train_pol, comp_Y_train))
        test_score_ridge = (ridge_pol.score(comp_X_test_pol, comp_Y_test))
        test_scores_ridge.append(test_score_ridge)
        train_scores_ridge.append(train_score_ridge)
        print('The train score is: ', train_score_ridge)
        print('The test score is: ', test_score_ridge)
        
        print('\033[1m'+'For the ElasticNet regression model : '+'\033[0m')
        regr = ElasticNet(random_state=0, alpha=1).fit(comp_X_train, comp_Y_train)
        train_score_elast = (regr.score(comp_X_train, comp_Y_train))
        test_score_elast = (regr.score(comp_X_test, comp_Y_test))
        test_scores_elasticNet.append(test_score_elast)
        train_scores_elasticNet.append(train_score_elast)
        print('The train score is: ', train_score_elast)
        print('The test score is: ', test_score_elast)
        
        print('')
    
    return test_scores, test_scores_ridge, test_scores_elasticNet, train_scores, train_scores_ridge, train_scores_elasticNet


def plot_regression_epochs(epochs, test_scores, train_scores, method_name) :
    '''
    PLot the training scores and testing scores in function of the epoch (can be a general plotting train/test function but mainly used 
    for the output of the above pipeline)
    Inputs :
        * epochs : maximum number of epochs
        * test_scores : test score at each epoch (e.g. R^2)
        * train_scores : train score at each epoch (e.g. R^2)
        * method_name : name of the model used to fit the data (e.g. linear regression) 
    '''
    
    #Turn epochs into an array [1, 2, ....., epochs]
    Epochs = np.arange(1, epochs+1)
    
    #PLot
    plt.figure(figsize=(9,5)) 
    plt.plot(Epochs, test_scores, 'ro', label = 'test')
    plt.plot(Epochs, train_scores, 'bo', label = 'train')
    plt.xlabel('epochs to train encoder') ; plt.ylabel('R^2')
    title = 'R^2 score in function of the epochs used to train the encoder using a ' + method_name
    plt.title(title)
    plt.legend()
    plt.savefig("Epochs_Regression with "+ method_name)
    plt.show()


def pipeline_ParametersQuality_subsample(dataset_name= 'middle_small', params_name= "Data/params_middle_small.csv.bz2", epochs=50, degree=3, alpha_=1,  learningRate = 1e-4, neuron_=10, criterion = nn.MSELoss(), seed=1) :
    
    '''
    This pipeline tries to fit a regression model to the compressed data (solution of the encoder) with the true parameters of 
    the haemodynamic problem for the different subsampled datasets.
    
    This pipeline is meant to do the following steps for each sub-dataset in the the folder 'dataset_name' :
        - Load the dataset
        - Split the dataset in x_train and x_test, same for the parameters, they will the "y" of our future regression problem
        - Train/Test the autoencoder for a certain number of epochs
        - Fit a ridge regression with plynomial extension. X = solution of the encoder (compression) and Y = real parameters of the problem
    Inputs:
        * dataset_name : name of the folder of the subdatasets
        * params_name : real parameters that generated this dataset
        * epochs : maximum number of epochs
        * degree : degree of polynomial extension
        * alpha_ : ridge regularization factor
        * learningRate : learning rate for the auto-encoder
        * neuron_ : number of neurons in the latent space
        * criterion : metric used for the optimization of the auto-encoder  (default MSE loss)
        * seed : fix the train/test split
    Outputs :
        * test_scores : Test scores of the ridge regression for each subdataset
        * train_scores : Train scores of the ridge regression for each subdataset
        * names : Names of each sub-datasets (just to keep track of them)
    '''
    
    file_location = os.path.join('data', 'pickle', dataset_name, '*')
    filenames = glob.glob(file_location)
    i=0 

    
    test_scores = []
    train_scores = []
    names = [] #Just to be sure of which error corresponds to which sub-dataset
    
    #Load the real parameters of the haemodynamic problem
    compressed_Y_pd = pd.read_csv(params_name, header=None)
    comp_Y = compressed_Y_pd.to_numpy()
    comp_Y_train, comp_Y_test = train_test_split(comp_Y, test_size=0.1, random_state=seed)

    for f in filenames:

        print("\033[1m" +'Train/Test of the sub-dataset ' + f + "\033[0m")

        #Load the data
        x  = cPickle.load(open(f, "rb"))

        #Split
        x_train, x_test = train_test_split(x, test_size=0.1, random_state=seed)
        y_train, y_test = x_train, x_test

        #Declare the model
        input_size=x_train.shape[1]
        model = Autoencoder(input_size, neuron_)
        optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=1e-5)

        #Dataloader
        train_loader = DataLoader(x_train)
        test_loader = DataLoader(x_test)

        #Training and testing over the epochs
        for epoch in range(epochs):
            # train the model 
            train_loss=train_epoch(train_loader, model, criterion, optimizer)
            # compute the relative training error
            train_error = valid_epoch(train_loader, model)
            # compute the relative test error
            test_error=valid_epoch(test_loader, model)

        print("\033[1m" +"Epoch:{}/{} Training Error:{:.3f} Test Error:{:.3f}".format(epoch + 1,epochs,train_error,test_error)+"\033[0m")
        
        #Extract the compression
        compressed_X_train = []
        for train in train_loader :
            train = train.float()
            compressed_x_train = model.forward(x=train, show_compression=True)
            compressed_X_train.append(compressed_x_train.detach().numpy())

        compressed_X_test = []    
        for test in test_loader :
            test = test.float()
            compressed_x_test = model.forward(x=test, show_compression=True)
            compressed_X_test.append(compressed_x_test.detach().numpy())       

        #Concatenate testing and training set
        comp_X_train = np.concatenate(np.array(compressed_X_train))
        comp_X_test = (np.squeeze(np.array(compressed_X_test)))

        #Polynomial extension
        exponents = np.arange(degree) + 1

        comp_X_train_pol = []
        for row in comp_X_train: 
            comp_X_train_pol.append((row**exponents[:,None]).ravel())

        comp_X_test_pol = []
        for row in comp_X_test: 
            comp_X_test_pol.append((row**exponents[:,None]).ravel())  

            
        #Regularized regression
        ridge_pol = Ridge(alpha=alpha_).fit(comp_X_train_pol, comp_Y_train)
        train_score = (ridge_pol.score(comp_X_train_pol, comp_Y_train))
        test_score = (ridge_pol.score(comp_X_test_pol, comp_Y_test))
        test_scores.append(test_score)
        train_scores.append(train_score)
        print('The train score is: ', train_score)
        print('The test score is: ', test_score)

        names.append(f)
        
    return test_scores, train_scores, names

