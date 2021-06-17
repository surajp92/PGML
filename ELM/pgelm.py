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
def u(x,ipr):
    if ipr == 1:
        y = sin(10*pi*x)/(2*x) + (x-1)**4 
    elif ipr == 2:
        y = (x - np.sqrt(2))*(np.sin(8.0*np.pi*x))**2
    elif ipr == 3:
        y = x**2 + np.sin(8.0*np.pi*x + np.pi/10.0)**2
    return y

def xPG(x,ipr):
    if ipr == 1:
        y = (x-1)**4
    elif ipr == 2:
        y = np.sin(8.0*np.pi*x)
    elif ipr == 3:
        y = np.sin(8.0*np.pi*x)
    #y = sin(10*pi*x)/(2*x) + (x-1)**4
    return y

def activation(x):
    F = 2/(1+exp(-2*x)) - 1
    #F = 1/(1+exp(-x))
    F = np.tanh(x)
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


def pgelm_train(X,Xpg,Y,Q,a_sc,b_sc,im):
    Nin, Ntrain = X.shape
    Npg, _ = Xpg.shape
    Nout, _ = Y.shape
    
    #Define ELM matrices

    #b = np.random.randn(Q)
    b = np.random.uniform(low=a_sc, high=b_sc, size = Q)
    B = np.transpose([b] * Ntrain)
    
    #C = np.random.randn(Q,Nin)
    C = np.random.uniform(low=a_sc, high=b_sc, size = (Q,Nin))
    
    Z = B + C @ X
    if im == 'pgelm':
        Z = np.vstack([Z,Xpg])
    H = activation(Z)
    H = H.T

    W = np.linalg.pinv(H) @ Y.T
    W = W.T
    
    err = np.linalg.norm(W @ H.T-Y)/np.sqrt(Ntrain)
        
    return b, C, W, err


def pgelm_pred(X,Xpg,b,C,W,im):
    
    Nin, Npred = X.shape

    B = np.transpose([b] * Npred)
    Z = B + C @ X
    if im == 'pgelm':
        Z = np.vstack([Z,Xpg])
    H = activation(Z)    
    H = H.T
    
    Y = W @ H.T
    
    return Y


#%%
# floag for model
im = 'pgelm' # [1] ELM, [2] PG-ELM
# paper (JCP2020): A composite neural network that learns from multi-fidelity data: 
#    Application to function approximation and inverse PDE problems
# Xuhui Meng,  George Em Karniadakis 2020 JCP

ipr = 2 # flag for problem [1] , [2] 3.1.3 (JCP2020), [3] 3.1.4 (JCP2020)

#Scale data
a_sc = -2
b_sc = 2

if ipr == 1:
    xlow, xhigh = 0.6, 2.4
elif ipr == 2:
    xlow, xhigh = 0, 1
elif ipr == 3:
    xlow, xhigh = 0, 1    

nens = 5
ypred = np.zeros([nens,512])

fig, ax = plt.subplots(1,1, figsize=(6,5))

for i in range(nens):
    
    seed_num = (i+1)*10
    print(seed_num)
    
    noise = 0.0    # noise-free  
    #noise = 0.05   # 5% noise
    
    #np.random.seed(seed=seed_num)  # fix the realization of Guassian white noise for sake of reproducing
  
    #training data
    Ntrain = 32    #number of training points
    xu= np.linspace(xlow,xhigh,Ntrain).reshape((1,-1))
    
    yu = u(xu,ipr) # training outputs for \Omega2 and \Omega 3 or observations y_2 and y_3
    yu = yu + noise * np.std(yu) * np.random.randn(1,Ntrain) # add the noise
    
        
    xPGu = xPG(xu,ipr) # training outputs for \Omega2 and \Omega 3 or observations y_2 and y_3
    
    #testing data
    xt = np.linspace(xlow,xhigh,512).reshape((1,-1))  #(np.concatenate((xx.reshape((-1,1)),tt.reshape((-1,1))),axis=1) # test inputs on lattice
    yt = u(xt,ipr)           # test outputs
    xPGt = xPG(xt, ipr)       
    


    #%%
    
    Q = 30 #number of neurons
    
    
    Xtrain = xu
    Xtrain_pg = xPGu
    Ytrain = yu
    
    Xmin, Xmax, Xtrain_sc = scale_train(Xtrain,a_sc,b_sc)
    Xmin_pg, Xmax_pg, Xtrain_pg_sc = scale_train(Xtrain_pg,a_sc,b_sc)
    Ymin, Ymax, Ytrain_sc = scale_train(Ytrain,a_sc,b_sc)
    
    #Xtrain_sc, Xtrain_pg_sc, Ytrain_sc = Xtrain, Xtrain_pg, Ytrain
    
    B, C, W, err = pgelm_train(Xtrain_sc,Xtrain_pg_sc,Ytrain_sc,Q,a_sc,b_sc,im)
    print('RMSE=', err)
    
    #%% Testing
    Xtest = xt
    Xtest_pg = xPGt
    Ytest = yt
    
    Xtest_sc = scale_test(Xtest,Xmin,Xmax,a_sc,b_sc)
    Xtest_pg_sc = scale_test(Xtest_pg,Xmin_pg,Xmax_pg,a_sc,b_sc)

    #Xtest_sc, Xtest_pg_sc = Xtest, Xtest_pg

    Ypred_sc = pgelm_pred(Xtest_sc,Xtest_pg_sc,B,C,W,im)
    Ypred = rescale(Ypred_sc,Ymin,Ymax,a_sc,b_sc)
    
    #Ypred = Ypred_sc
    #%%
    
    
    ax.plot(Xtest[0],Ypred[0],'--',label=f'Ensemble_{i}')

ax.plot(Xtest[0],Ytest[0],'k',label='True')
# ax.set_ylim([1.25*np.min(),5])
ax.legend()
fig.savefig(f'prediction_{ipr}_{im}.png', dpi=200)

    