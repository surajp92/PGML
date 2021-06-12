#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 15:26:04 2021

@author: suraj
"""

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from keras import backend as kb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.regularizers import l2

from tensorflow.keras import regularizers


#from helpers import get_lift_drag, preprocess_image, read_csv_file, save_model, load_model

# font for plotting
import matplotlib as mpl
font = {'family' : 'normal',
        'size'   : 14}
mpl.rc('font', **font)

#%%

font = {'size'   : 14}    
plt.rc('font', **font)

filename = f'../../../Similarity_Solution/su2.csv'
data_blasisus = np.genfromtxt(filename, delimiter=',')

data = np.load('./Re_0.5e+5/data.npz')

u = data['u']
v = data['v']
p = data['p']
x = data['x']
y = data['y']
X = data['X']
Y = data['Y']

nx = x.shape[0]
ny = y.shape[0]

Xf = X.flatten()
Yf = Y.flatten()

uf = u.flatten()
vf = v.flatten()
pf = p.flatten()

#%%
features = np.vstack((Xf, Yf)).T
labels = np.array([uf,vf,pf]).T

sampling = 2 # [1] half-x, [2] random 50%, 

if sampling == 1:
    train_slice = features[:,0] <= 0.05
    features_train = features[train_slice]
    labels_train = labels[train_slice]
elif sampling == 2:
    num_samples = features.shape[0]
    fraction = 0.5
    train_slice = np.random.randint(num_samples, size=int(fraction*num_samples))
    features_train = features[train_slice]
    labels_train = labels[train_slice]
    
#%%
sc_input = MinMaxScaler(feature_range=(-1,1))
sc_input = sc_input.fit(features_train)
xtrain = sc_input.transform(features_train)

sc_output = MinMaxScaler(feature_range=(-1,1))
sc_output = sc_output.fit(labels_train)
ytrain = sc_output.transform(labels_train)

#%%
nf1 = 2
nl = 3

n_layers = 2
n_neurons = [20,20]
lr = 0.001
epochs = 600
batch_size = 64

x1 = Input(shape=(nf1,))

x = Dense(n_neurons[0],activation='relu')(x1)
x = Dense(n_neurons[0],activation='relu')(x)

output = Dense(nl,activation='linear')(x)

model = Model(inputs=x1, outputs=output)
    
print(model.summary())
opt = tf.keras.optimizers.Adam(learning_rate=lr)

"""## compile the model"""
model.compile(loss='mean_squared_error', optimizer=opt)

history_callback = model.fit(x=xtrain, 
                             y=ytrain, 
                             batch_size=batch_size, 
                             epochs=epochs, 
                             validation_split=0.3, 
                             shuffle=True, 
                             callbacks=[])

#%%
loss = history_callback.history["loss"]
val_loss = history_callback.history["val_loss"]

plt.figure()
epochs = range(1, len(loss) + 1)
plt.semilogy(epochs, loss, 'b', label='Training loss')
plt.semilogy(epochs, val_loss, 'r', label='Validationloss')
plt.title('Training and validation loss')
plt.legend()
filename = f'loss.png'
plt.savefig(filename, dpi = 300)
plt.show()

#%%
xtest = sc_input.transform(features)
ytest = np.copy(labels)

ypred_sc = model.predict(xtest)
ypred = sc_output.inverse_transform(ypred_sc)

#%%
x = data['x']
y = data['y']
n_slice = 200
x_slice = features[:,0] == x[n_slice]
x_005 = features[x_slice]

#%%
print(f'Prediction for x = {x[n_slice]}')
ytest_x = ytest[x_slice]
ypred_x = ypred[x_slice]

#%%
fig, ax = plt.subplots(1,1,figsize=(6,5),sharey=True,)

ax.plot(ytest_x[:,0],y,'ro-',label='True')
ax.plot(ypred_x[:,0],y,'bs-',label='Pred')
ax.set_ylim([0,0.003])
ax.legend()
ax.grid()
plt.show()

#%%
fig, ax = plt.subplots(3,1,figsize=(8,12),sharey=True,)
cs = ax[0].contourf(X,Y, np.reshape(ytest[:,2],[ny,nx]),120)
fig.colorbar(cs, ax=ax[0])

cs = ax[1].contourf(X,Y, np.reshape(ypred[:,2],[ny,nx]),120)
fig.colorbar(cs, ax=ax[1])

diff = ytest - ypred

cs = ax[2].contourf(X,Y, np.reshape(diff[:,2],[ny,nx]),120, cmap='jet')
fig.colorbar(cs, ax=ax[2])

for i in range(3):
    ax[i].set_ylim([0,0.004])

plt.show()