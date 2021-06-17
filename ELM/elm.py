# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:26:26 2021

@author: Shady
"""


#%% Import libraries
import numpy as np   
from numpy import pi, sin, cos, exp
import matplotlib.pyplot as plt
import time
import os


#%%
def u(x):
    y = sin(10*pi*x)/(2*x) + (x-1)**4 
    return y

def xPG(x):
    y = (x-1)**4
    y = sin(10*pi*x)/(2*x) + (x-1)**4
    return y

def activation(x):
    # F = 2/(1+exp(-2*x)) - 1
    F = 1/(1+exp(-x))
    # F = np.tanh(x)
    return F

def scale_train(x,a,b):
    xmin = np.min(x,axis = 1)
    xmax = np.max(x,axis = 1)
    xsc = a + (x-xmin)*(b-a)/(xmax-xmin)
    return xmin, xmax, xsc
  
def scale_test(x,xmin,xmax,a,b):
    xsc = a + (x-xmin)*(b-a)/(xmax-xmin)
    return xsc
      

def rescale(xsc,xmin,xmax,a,b):
    x = (xsc - a)*(xmax-xmin)/(b-a) + xmin
    return x


def elm_train(X,Y,Q,a_sc,b_sc):
    Nin, Ntrain = X.shape
    Nout, _ = Y.shape
    
    #Define ELM matrices

    #b = np.random.randn(Q)
    b = np.random.uniform(low=a_sc, high=b_sc, size = Q)
    B = np.transpose([b] * Ntrain)
    
    #C = np.random.randn(Q,Nin)
    C = np.random.uniform(low=a_sc, high=b_sc, size = (Q,Nin))
    
    Z = B + C @ X
    H = activation(Z)
    H = H.T

    W = np.linalg.pinv(H) @ Y.T
    W = W.T
    
    err = np.linalg.norm(W @ H.T-Y)/np.sqrt(Ntrain)
    
    return b, C, W, err


def elm_pred(X,b,C,W):
    
    Nin, Npred = X.shape

    B = np.transpose([b] * Npred)
    Z = B + C @ X
    H = activation(Z)
    H = H.T
    
    Y = W @ H.T
    
    return Y


#%%
nens = 5
ypred = np.zeros([nens,512])

for i in range(nens):
    
    seed_num = (i+1)*10
    print(seed_num)
    
    noise = 0.0    # noise-free  
    #noise = 0.05   # 5% noise
    
    np.random.seed(seed=seed_num)  # fix the realization of Guassian white noise for sake of reproducing
  
    #training data
    Ntrain = 128 #number of training points
    xu= np.linspace(0.6,2.4,Ntrain).reshape((1,-1))
    
    yu = u(xu) # training outputs for \Omega2 and \Omega 3 or observations y_2 and y_3
    yu = yu + noise * np.std(yu) * np.random.randn(1,Ntrain) # add the noise
    
        
    xPGu = xPG(xu) # training outputs for \Omega2 and \Omega 3 or observations y_2 and y_3
    
    #testing data
    xt = np.linspace(0.6,2.4,512).reshape((1,-1))  #(np.concatenate((xx.reshape((-1,1)),tt.reshape((-1,1))),axis=1) # test inputs on lattice
    yt = u(xt)           # test outputs
    xPGt = xPG(xt)       
    


    #%%
    
    Q = 30 #number of neurons
    
    
    Xtrain = xu
    Ytrain = yu
    
    #Scale data
    a_sc = -2
    b_sc = 2
    Xmin, Xmax, Xtrain_sc = scale_train(Xtrain,a_sc,b_sc)
    Ymin, Ymax, Ytrain_sc = scale_train(Ytrain,a_sc,b_sc)
    
    
    B, C, W, err = elm_train(Xtrain_sc,Ytrain_sc,Q,a_sc,b_sc)
    print('RMSE=', err)
    
    #%% Testing
    Xtest = xt
    Ytest = yt
    Ntest = 512
    
    Xtest_sc = scale_test(Xtest,Xmin,Xmax,a_sc,b_sc)
    
    Ypred_sc = elm_pred(Xtest_sc,B,C,W)
    
    Ypred = rescale(Ypred_sc,Ymin,Ymax,a_sc,b_sc)
    
    #%%
    
    plt.plot(Xtest[0],Ytest[0])
    plt.plot(Xtest[0],Ypred[0],'--')