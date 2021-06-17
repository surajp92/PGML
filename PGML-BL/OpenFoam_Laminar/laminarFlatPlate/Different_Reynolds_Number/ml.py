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

## grid is the central object in VTK where every field is added on to grid

grid_5 = pv.UnstructuredGrid(f'./Re_0.5e+5/VTK/Re_0.5e+5_10000.vtk')

centers = grid_5.cell_centers()

cell_centes = np.array(centers.points)

U_cell = np.array(grid_5.cell_arrays['U'])
P_cell = np.array(grid_5.cell_arrays['p'])

#%%
scale = 0.001
features = cell_centes[:,:-1]/scale
labels = np.hstack((U_cell[:,:-1], P_cell.reshape([-1,1])))

#%%
x_lims_low = features[:,0] >= 0
x_lims_high = features[:,0] <= 8
x_lims = x_lims_low == x_lims_high


#%%
features_xc = features[x_lims]

y_lims = features_xc[:,1] <= 1

features_c = features_xc[y_lims]

labels_xc = labels[x_lims]
labels_c = labels_xc[y_lims]

#%%
sc_input = MinMaxScaler(feature_range=(-1,1))
sc_input = sc_input.fit(features_c)
xtrain = sc_input.transform(features_c)

sc_output = MinMaxScaler(feature_range=(-1,1))
sc_output = sc_output.fit(labels_c)
ytrain = sc_output.transform(labels_c)

#%%
nf1 = 2
nl = 3

n_layers = 2
n_neurons = [20,20]
lr = 0.001
epochs = 800
batch_size = 16

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
xtest = sc_input.transform(features_c)
ytest = np.copy(labels_c)

ypred_sc = model.predict(xtest)
ypred = sc_output.inverse_transform(ypred_sc)

#%%
x_005_slice = features_c[:,0] == features_c[5,0]
x_005 = features_c[x_005_slice]
ytest_x005 = ytest[x_005_slice]
ypred_x005 = ypred[x_005_slice]

#%%
fig, ax = plt.subplots(1,1,figsize=(6,5),sharey=True,)

ax.plot(ytest_x005[:,0],x_005[:,1],'ro-')
ax.plot(ypred_x005[:,0],x_005[:,1],'bs-')

ax.grid()
plt.show()