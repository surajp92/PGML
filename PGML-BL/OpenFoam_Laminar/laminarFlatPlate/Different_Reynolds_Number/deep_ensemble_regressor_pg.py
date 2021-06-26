#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:54:36 2021

@author: suraj
"""

import numpy as np
import keras
from numpy.random import seed
seed(10)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import concatenate
import tensorflow.keras.backend as K
import os

from keras import backend as kb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.regularizers import l2


import matplotlib.pyplot as plt

import tensorflow as tf
tf.python.framework_ops.disable_eager_execution()

def deep_ensemble_regression_nll_loss(sigma_sq, epsilon = 1e-6):
    """
        Regression loss for a Deep Ensemble, using the negative log-likelihood loss.
        This function returns a keras regression loss, given a symbolic tensor for the sigma square output of the model.
        The training model should return the mean, while the testing/prediction model should return the mean and variance.4
    """
    def nll_loss(y_true, y_pred):
        return 0.5 * K.mean(K.log(sigma_sq + epsilon) + K.square(y_true - y_pred) / (sigma_sq + epsilon))

    return nll_loss

def toy_dataset(input):
    output = []

    for inp in input:
        std = 3 if inp < 0 else 1
        out = [inp ** 3 + np.random.normal(0, std), 10*np.sin(inp)  + np.random.normal(0, std)]
        output.append(out)

    return np.array(output)

def mlp_model():
    inp = Input(shape=(2,))
    inp_pg = Input(shape=(1,))
    
    x = Dense(20, activation="relu")(inp)
    x = Dense(20, activation="relu")(x)
    
    x = concatenate(inputs=[x, inp_pg])
    
    x = Dense(20, activation="relu")(x)

    mean = Dense(3, activation="linear")(x)
    var = Dense(3, activation="softplus")(x)

    train_model = Model(inputs=[inp,inp_pg], outputs=mean)
    pred_model = Model(inputs=[inp,inp_pg], outputs=[mean, var])

    train_model.compile(loss=deep_ensemble_regression_nll_loss(var), 
                        optimizer="adam")

    return train_model, pred_model

filename = f'../../../Similarity_Solution/su2.csv'
data_blasisus = np.genfromtxt(filename, delimiter=',')

data = np.load('./Re_0.5e+5/data.npz')

noscale = False
adv_train = True

u = data['u']
v = data['v']
p = data['p']
x = data['x']
y = data['y']
X = data['X']
Y = data['Y']

#%%
plt.plot(u[:,5], y, label=f'x = {x[5]}')
plt.plot(u[:,-1], y, '--', label=f'x = {x[-1]}')
plt.ylim([0,0.002])
plt.legend()
plt.show()

#%%
filename = f'../../../Similarity_Solution/blasius_solution.npz'
data_blasisus = np.load(filename)

eta = data_blasisus['eta']
uu = data_blasisus['uu']
uinf = 0.5
nu = 1.0e-6

upg = np.zeros_like(u)

i = 0
upg[:,i] = u[:,i]

for i in range(1,x.shape[0]):
    etai = y*np.sqrt(uinf/(nu*x[i]))
    uuinterp = np.interp(etai, eta, uu)
    upg[:,i] = uuinterp*uinf
    
plt.plot(u[:,5], y, label=f'x = {x[5]} CFD')
plt.plot(upg[:,5], y, '--', label=f'x = {x[5]} Blasius')
plt.ylim([0,0.002])
plt.legend()
plt.show()   

#%%
nx = x.shape[0]
ny = y.shape[0]

Xf = X.flatten()
Yf = Y.flatten()

uf = u.flatten()
vf = v.flatten()
pf = p.flatten()

upgf = upg.flatten()

#%%
features = np.vstack((Xf, Yf)).T
features_pg = upgf.reshape([-1,1])
labels = np.array([uf,vf,pf]).T

sampling = 2 # [1] half-x, [2] random 50%, 

if sampling == 1:
    train_slice = features[:,0] <= 0.05
    features_train = features[train_slice]
    features_pg_train = features_pg[train_slice]
    labels_train = labels[train_slice]
elif sampling == 2:
    num_samples = features.shape[0]
    fraction = 0.5
    train_slice = np.random.randint(num_samples, size=int(fraction*num_samples))
    features_train = features[train_slice]
    features_pg_train = features_pg[train_slice]
    labels_train = labels[train_slice]

#labels_train = labels[train_slice] #+ 0.1*np.random.randn(labels_train.shape[0], labels_train.shape[1])    

#%%
if noscale:
    xtrain = features_train
    xtrain_pg = features_pg_train
    ytrain = labels_train
else:
    sc_input = MinMaxScaler(feature_range=(-1,1))
    sc_input = sc_input.fit(features_train)
    xtrain = sc_input.transform(features_train)
    
    sc_pg = MinMaxScaler(feature_range=(-1,1))
    sc_pg = sc_pg.fit(features_pg_train)
    xtrain_pg = sc_pg.transform(features_pg_train)
    
    sc_output = MinMaxScaler(feature_range=(-1,1))
    sc_output = sc_output.fit(labels_train)
    ytrain = sc_output.transform(labels_train)
    
    if adv_train:
        ytrain = ytrain + 0.05*np.random.randn(ytrain.shape[0], ytrain.shape[1])


#%%
num_estimators = 5

train_estimators = [None] * num_estimators 
test_estimators = [None] * num_estimators

for i in range(num_estimators):        
    train_estimators[i], test_estimators[i] = mlp_model()

folder = 'regression-ens-2f-pg'
if not os.path.exists(folder):
        os.makedirs(folder)

#%%        
#    train_model, pred_model = DeepEnsembleRegressor(mlp_model, 5)
for i in range(num_estimators):
    train_estimators[i].fit(x=[xtrain, xtrain_pg], y=ytrain, epochs=200)
    filename = os.path.join(folder, f'model-ensemble-{i}.hdf5')
    test_estimators[i].save(filename)

#%%
if noscale:
    xtest = features
else:    
    xtest = sc_input.transform(features)
    xtest_pg = sc_pg.transform(features_pg)

ytest = np.copy(labels)

means = []
variances = []

for i in range(num_estimators):
    mean, variance = test_estimators[i].predict(x=[xtest, xtest_pg])
#    if noscale:
#        mean = np.copy(mean)
#    else:
#        mean = sc_output.inverse_transform(mean)
    means.append(mean)
    variances.append(variance)

#%%
means = np.array(means)
variances = np.array(variances)

#%%
mixture_mean = np.mean(means, axis=0)
mixture_var  = np.mean(variances + np.square(means), axis=0) - np.square(mixture_mean)
mixture_var[mixture_var < 0.0] = 0.0
    
y_pred_mean, y_pred_std = mixture_mean, np.sqrt(mixture_var)

print("y pred mean shape: {}, y_pred_std shape: {}".format(y_pred_mean.shape, y_pred_std.shape))

y_pred_up_1 = y_pred_mean + y_pred_std
y_pred_down_1 = y_pred_mean - y_pred_std

y_pred_up_2 = y_pred_mean + 2.0 * y_pred_std
y_pred_down_2 = y_pred_mean - 2.0 * y_pred_std

y_pred_up_3 = y_pred_mean + 3.0 * y_pred_std
y_pred_down_3 = y_pred_mean - 3.0 * y_pred_std

y_pred_mean = sc_output.inverse_transform(y_pred_mean)

y_pred_up_1 = sc_output.inverse_transform(y_pred_up_1)
y_pred_down_1 = sc_output.inverse_transform(y_pred_down_1)

y_pred_up_2 = sc_output.inverse_transform(y_pred_up_2)
y_pred_down_2 = sc_output.inverse_transform(y_pred_down_2)

y_pred_up_3 = sc_output.inverse_transform(y_pred_up_3)
y_pred_down_3 = sc_output.inverse_transform(y_pred_down_3)

#%%
x = data['x']
y = data['y']
n_slice = 100
x_slice = features[:,0] == x[n_slice]
x_005 = features[x_slice]

#%%
x_slice_train = features_train[:,0] == x[n_slice]
x_005_train = features_train[x_slice_train]

ytrain_x = ytrain[x_slice_train]
ytrain_x = sc_output.inverse_transform(ytrain_x)

#%%
print(f'Prediction for x = {x[n_slice]}')
ytest_x = ytest[x_slice]
ypred_x = y_pred_mean[x_slice]
ypred_x_up = y_pred_up_1[x_slice]
ypred_x_down = y_pred_down_1[x_slice]

#%%
fig, ax = plt.subplots(1,1,figsize=(6,5),sharey=True,)

plot_var = 1

ax.plot(x_005_train[:,1],ytrain_x[:,plot_var],'ko-',
        ls='none',fillstyle='none',label='Train')

ax.plot(y,ytest_x[:,plot_var],'r-',lw=2,label='True')
ax.plot(y,ypred_x[:,plot_var],'b-',lw=2,label='Pred')
ax.fill_between(y,ypred_x_down[:,plot_var], ypred_x_up[:,plot_var],
                color=(0, 0, 0.9, 0.5), 
                label="One Sigma Confidence Interval")

ax.set_xlim([0,0.003])
ax.legend()
ax.grid()
plt.show()
fig.tight_layout()
filename = os.path.join(folder, 'deep_ensemble_prediction_pg.png')
fig.savefig(filename, dpi=200)
