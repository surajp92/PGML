"""
Created on Wed Jun 16 21:43:41 2021

@author: suraj
"""
import numpy as np   
import pandas as pd
from numpy import pi, sin, cos, exp
import matplotlib.pyplot as plt
import time
import os
from sklearn.preprocessing import MinMaxScaler

def training_data_cl_cd(data,panel_data,aoa,re,lift,drag):
    '''
    Input:
    data ---- this contains the x and y coordinate of airfoil
    panel_data ---- this contains the CL and CD_p determined using the panel method
    aoa ---- angle of attack
    re ---- Reynolds number
    lift ---- lift coefficient from XFoil 
    drag ---- drag coefficient from XFoil
    
    Output:
    xtrain ---- input to the neural network for training 
                (airfoil shape, CL and CD_p from panel method, aoa, re)
    ytrain ---- label of the training dataset (Xfoil lift coefficient)    
    '''
    num_samples = data.shape[0]
    npoints = data.shape[1]
    num_cp = npoints - 1
    nf = data.shape[2]
    xtrain = np.zeros((num_samples,npoints*nf+2+2))
    ytrain = np.zeros((num_samples,1))
    
    for i in range(num_samples):
        xtrain[i,:npoints] = data[i,:,0]
        xtrain[i,npoints:2*npoints] = data[i,:,1]
        xtrain[i,-4] = panel_data[i,0]
        xtrain[i,-3] = panel_data[i,1]
        xtrain[i,-2] = aoa[i]
        xtrain[i,-1] = re[i]
        
        ytrain[i,0] = lift[i]
#        ytrain[i,1] = drag[i]
    
    return xtrain, ytrain

def activation(x):
    F = 2/(1+exp(-2*x)) - 1
    #F = 1/(1+exp(-x))
    F = np.tanh(x)
    return F


def scale_train(x,a,b):
    xmin = np.min(x,axis = 1).reshape(-1,1)
    xmax = np.max(x,axis = 1).reshape(-1,1)
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

im = 'elm' # [1] ELM, [2] PG-ELM

# number of cordinates defining the airfoil shape
num_xy = 201
num_cp = num_xy - 1

a_sc = -1
b_sc = 1

nens = 10
Q = 30

data = np.load('../train_data_re.npz')
data_xy = data['data_xy']
panel_data = data['panel_data']
aoa = data['aoa']
re = data['re']
cl = data['cl']
cd = data['cd']
cm = data['cm']

xtrain, ytrain = training_data_cl_cd(data_xy,panel_data,aoa,re,cl,cd)

sc_input = MinMaxScaler(feature_range=(a_sc,b_sc))
sc_input = sc_input.fit(xtrain)
xtrain_sc = sc_input.transform(xtrain)

sc_output = MinMaxScaler(feature_range=(a_sc,b_sc))
sc_output = sc_output.fit(ytrain)
ytrain_sc = sc_output.transform(ytrain)


xtrain, ytrain = xtrain_sc, ytrain_sc

# only shape of the airfoil (x and y cordinates)
Xtrain_sc = np.copy(xtrain[:,:2*num_xy].T)

# aoa, re, and physics-based features from panel method
Xtrain_pg_sc = np.copy(xtrain[:,2*num_xy:].T)

Ytrain_sc = ytrain_sc.T

#%%
data = np.load('../test_data_re_23024.npz')
airfoil_names_test = data['airfoil_names_test']
data_xy_test = data['data_xy_test']
panel_data_test = data['panel_data_test']
aoa_test = data['aoa_test']
re_test = data['re_test']
cl_test = data['cl_test']
cd_test = data['cd_test']
cm_test = data['cm_test']

xtest, ytest = training_data_cl_cd(data_xy_test,panel_data_test,aoa_test,re_test,cl_test,cd_test)

# scale the test data
xtest_sc = sc_input.transform(xtest[:,:])  
Xtest_sc = np.copy(xtest_sc[:,:2*num_xy].T) # airfol shape features
Xtest_pg_sc = np.copy(xtest_sc[:,2*num_xy:].T) # physics-based features features        


#%%

fig, ax = plt.subplots(1,1, figsize=(6,5))

for i in range(nens):
    
    seed_num = (i+1)*10
    print(seed_num)
    
    B, C, W, err = pgelm_train(Xtrain_sc,Xtrain_pg_sc,Ytrain_sc,Q,a_sc,b_sc,im)
    print('RMSE=', err)
    
    Ypred_sc = pgelm_pred(Xtest_sc,Xtest_pg_sc,B,C,W,im)
    ypred = sc_output.inverse_transform(Ypred_sc)
    
    ax.plot(aoa_test, ypred.T, label=f'ML-{i} NACA23024')

ax.plot(aoa_test, ytest[:,0], 'ks-', label=f'Xfoil NACA23024')    
ax.legend()
plt.show()
fig.savefig(f'prediction_airfoil_{im}.png', dpi=200)
