# Machine Learning - Project 2 

## *Unsteady parametrized Stokes equations in a 2D arterial bifurcation with stenosis: design of an Autoencoder for data compression*

## General information 

The repository contains the code for the EPFL Machine Learning course (CS-433) 2021 for the __ML4Science__ project. This project is done in the context of the Partial Differential Equation with Deep Neural Networks (PDE-DNN) project run by Professor's Deparis lab. Using preexisting code, we have simulated the blood flow in an arterial bifurcation featuring a stenosis relying on Stokes equation and finite elements method with MATLAB (code not in the repository). Our goal is to implement an autoencoder (AE) to compress the mathematical solutions of the blood flow which are differentially sampled in time and space. In this work we make progress to this end by proposing two autoencoder (AE) architectures to identify and evolve a __low-dimensional__ representation of a spatiotemporal system. In particular, we employ PyTorch _Feed-Forward AEs_ and _Long short-term memory AEs_ (_LSTM AEs_) to learn an optimal low-dimensional representation of the full state of the system. Ideally, we would expect the compression to lay in a 5 dimensional subspace. Indeed, we know that there are exactly 5 physical parameters that characterize the equations, so theoretically it should be possible to have only 5 neurons in the latent space.  In a second time, we investigate the relationship between the low number of “abstract parameters” and the 5 initial physical parameters. This relationship would enable us to recover the 5 initial parameters and therefore the general solution by knowing only a subsample of this solution.

### Team members
The project is accomplished by the team `TheVipers` with members:

- Camille Frayssinhes: [@camillefrayssinhes](https://github.com/camillefrayssinhes)
- Assia Ouanaya: [@assiaoua](https://github.com/assiaoua)
- Theau Vannier: [@theauv](https://github.com/theauv)

### Data
The data used in our work describes the __blood flow__ in an artery with a bifurcation and a stenosis, which is modelled as the union of two semi-elliptical plaques. We generate the dataset by running on MatLab numerical simulations, which solve unsteady Stokes equations with random parameters characterizing the stenonis' shape. The output of each simulation is the discrete approximation of the time-evolution of the blood velocity u and pressure p in the considered domain, which is discretized with a suitable mesh-grid. There are 5 parameters that can be set: µ<sub>1</sub> stenosis width, µ<sub>2</sub> stenosis height, µ<sub>3</sub> distance between the 2 stenosis plaque centers, R<sub>1</sub> and R<sub>2</sub> resistive term of the first and of the second bifurcation respectively. We generate our dataset performing 175 simulations corresponding to randomly selected combinations of the 5 model parameters.  
<p align="center">
  <img src="/img/nice_illustration.png" alt="params" width="500"/>
</p> 

The data simulator generates 4 files:  
- u1.csv: which represents the x coordinate of the blood speed in the artery
- u2.csv: which represents the y coordinate of the blood speed in the artery
- p.csv: which represents the blood pressure in the artery
- params.csv: which represents the model parameters


***
## Project architecture

### Helper functions

`helpers.py`: loading CSV training and testing data, spatial visualization of the arterial bifurcation data points and some pipelines code used in the main (especially in the last part : physical interpretation).

### Processing data 

`preprocessing.py`: preprocessing training and test data for model prediction.

### AE models

`autoencoder.py`: feed-forward autoencoder implementation.

`LSTM.py`: LSTM autoencoder implementation.

### Selecting model

`crossvalidation.py`: using cross-validation to search for the best parameters(lambda & number latent neurons) to obtain the best test errors.

`crossvalidation_lstm.py`: using cross-validation to search for the best parameters(lambda, momentum & hidden size) to obtain the best test errors.


### Notebook

`main.ipynb`: data exploration and preprocessing. Deployment of Feed-Forward Autoencoder by tuning the best parameters through cross validation. Analysis and visualisation of the test and training errors with different choices of parameters. Linear regression between the abstract parameters of the latent space of the AE and the initial 5 physical parameters.

`lstm_main.ipynb`: data exploration and preprocessing. Deployment of Recurrent LSTM Autoencoder by tuning the best parameters through cross validation. Analysis and visualisation of the test and training errors with different choices of parameters.


### Report

`documents/report.pdf`: a 4-pages report of the project.


***
## AE architecture

### Feed-forward AE

The feed-forward AE consists of fully-connected linear layers stacked on top of each other. The input size - which is the same as the output size - depends on the sampling which is applied on time and space. Different latent sizes are tested during the training in `main.ipynb`.

<p align="center">
  <img src="/img/Architecture.png" alt="Feed_Architecture" width="500"/>
</p>

### LSTM AE

The architecture of the LSTM AE is displayed below. Again, the input and the output size depend on the sampling and different latent sizes are tested during the training in `lstm_main.ipynb`. Each layer of the encoder and of the decoder is a LSTM layer.

<p align="center">
  <img src="/img/Architecture_lstm.png" alt="Architecture_lstm" width="500"/>
</p>

***


## How to run the code
The project has been developed and tested with `python3.8`.
The required library for running the models and training is `numpy1.20.1`.
The library for visualization is `matplotlib3.3.4`.
The library for autoencoder implementation is `pytorch1.10.0`. 
Annex libraries used are `pandas`, `sklearn`, `os`, `glob` and `_pickle`.

All the data required to reproduce our results are stored in a drive, you can find them using this [link](https://drive.google.com/drive/folders/1mePpM9uwHIQnvnxh0rlZkAbvd7gRzQW4?usp=sharing).

From this drive repository, you have to download the 'data' folder in the folder of this project, namely 'ml-project-2-thevipers2'.

Then you can run the notebooks 'main' or 'lstm_main' . If you don't want to reload the whole dataset (bz2 files), you can directly jump to the pickle loading in the notebook (you should see a link in the notebook after the importations to do it).

***

## References

[1] Dal Santo, N., Deparis, S., and Pegolotti, L., “Data driven approximation of parametrized PDEs by reduced basis and neural networks”, <i>Journal of Computational Physics</i>, vol. 416, 2020. doi:10.1016/j.jcp.2020.109550.

