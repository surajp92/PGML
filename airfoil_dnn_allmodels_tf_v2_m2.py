#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:01:23 2020

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from panel_allairfoil import panel

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

import matplotlib as mpl
font = {'family' : 'normal',
        'size'   : 14}
mpl.rc('font', **font)

#%%
def coeff_determination(y_true, y_pred):
    SS_res =  kb.sum(kb.square(y_true-y_pred ))
    SS_tot = kb.sum(kb.square( y_true - kb.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + kb.epsilon()) )

def training_data(data,panel_data,aoa,re,lift,drag):
    num_samples = data.shape[0]
    npoints = data.shape[1]
    num_cp = npoints - 1
    nf = data.shape[2]
    xtrain = np.zeros((num_samples,npoints*nf+2))
    ytrain = np.zeros((num_samples,1))
    
    for i in range(num_samples):
        xtrain[i,:npoints] = data[i,:,0]
        xtrain[i,npoints:2*npoints] = data[i,:,1]
        xtrain[i,-2] = aoa[i]
        xtrain[i,-1] = re[i]
        
        ytrain[i,0] = lift[i]
#        ytrain[i,1] = drag[i]
    
    return xtrain, ytrain

def training_data_cp(data,panel_data,aoa,re,lift,drag):
    num_samples = data.shape[0]
    npoints = data.shape[1]
    num_cp = npoints - 1
    nf = data.shape[2]
    xtrain = np.zeros((num_samples,npoints*nf+num_cp+2))
    ytrain = np.zeros((num_samples,1))
    
    for i in range(num_samples):
        xtrain[i,:npoints] = data[i,:,0]
        xtrain[i,npoints:2*npoints] = data[i,:,1]
        xtrain[i,2*npoints:2*npoints+num_cp] = panel_data[i,2:]
        xtrain[i,-2] = aoa[i]
        xtrain[i,-1] = re[i]
        
        ytrain[i,0] = lift[i]
#        ytrain[i,1] = drag[i]
    
    return xtrain, ytrain

def training_data_cl_cd(data,panel_data,aoa,re,lift,drag):
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

def training_data_cl(data,panel_data,aoa,re,lift,drag):
    num_samples = data.shape[0]
    npoints = data.shape[1]
    num_cp = npoints - 1
    nf = data.shape[2]
    xtrain = np.zeros((num_samples,npoints*nf+2+1))
    ytrain = np.zeros((num_samples,1))
    
    for i in range(num_samples):
        xtrain[i,:npoints] = data[i,:,0]
        xtrain[i,npoints:2*npoints] = data[i,:,1]
        xtrain[i,-3] = panel_data[i,0]
        xtrain[i,-2] = aoa[i]
        xtrain[i,-1] = re[i]
        
        ytrain[i,0] = lift[i]
#        ytrain[i,1] = drag[i]
    
    return xtrain, ytrain

#%%
#input_shape = (64,64,1)
#cd_scale_factor = 10
#X1, X2, Y = read_csv_file(input_shape, data_version="v1", cd_scale_factor=10)

im = 2 # 1: (x,y), 2: (x,y,cl,cd), 3: (x,y,Cp) 4: (x,y,cl)

data_path = '../airfoil_shapes_re_v2/'
#data_path = '../airfoil_shapes/'

num_xy = 201
num_cp = num_xy - 1
  
db = data_path + 'train_data.csv'
df = pd.read_csv(db, encoding='utf-8')

col_name = []
col_name.append('Airfoil')
for i in range(num_xy):
    col_name.append(f'x{i}')
for i in range(num_xy):
    col_name.append(f'y{i}')
col_name.append('AOA1')    
col_name.append('AOA2')
col_name.append('RE')
col_name.append('CL')
col_name.append('CD')
col_name.append('CM')
col_name.append('CPmin')

df.columns = col_name

df = df[df['CL'].notna()]

#bad_airfoils = ['naca2206','naca4002','naca4404','naca4406','naca4602','naca6002','naca6004','naca6202','naca24002','naca24006','naca21004','naca22002','naca25002']
#df = df[~(df["Airfoil"].isin(bad_airfoils))]

num_samples = df.shape[0]

panel_data = np.zeros((num_samples,num_cp+2))
data_xy = np.zeros((num_samples,num_xy,2))

airfoil_names = []
aoa = np.zeros(num_samples)
re = np.zeros(num_samples)
cl = np.zeros(num_samples)
cd = np.zeros(num_samples)
cm = np.zeros(num_samples)

#%%
airfoils_xcoords = df.iloc[:,1:202].values
airfoils_ycoords = df.iloc[:,202:403].values

params = df.iloc[:,403:].values

#%%
counter = 0
cd_scale_factor = 10.0

generate_new_train_data = False

if generate_new_train_data :
    for index, row in df.iterrows():
        airfoil_names.append(row.Airfoil)
        aoa[counter] = row.AOA2
        re[counter] = row.RE
        cl[counter] = row.CL
        cd[counter] = row.CD*cd_scale_factor
        cm[counter] = row.CM
        
        data_xy[counter,:,0] = airfoils_xcoords[counter,:]
        data_xy[counter,:,1] = airfoils_ycoords[counter,:]
        
        CL, CDP, Cp, pp = panel(data_xy[counter,:,:], alfader=row.AOA2)
        panel_data[counter,0], panel_data[counter,1], panel_data[counter,2:] = CL,CDP, Cp
        
        print(counter, row.Airfoil, row.AOA2, row.RE)
        counter +=1

    aa,bb = np.where(panel_data < -35)
    aau = np.unique(aa)
    
    aoa = np.delete(aoa, aau, axis=0)
    re = np.delete(re, aau, axis=0)
    cl = np.delete(cl, aau, axis=0)
    cd = np.delete(cd, aau, axis=0)
    cm = np.delete(cm, aau, axis=0)
    
    data_xy = np.delete(data_xy, aau, axis=0)
    panel_data = np.delete(panel_data, aau, axis=0)
    
    num_samples = panel_data.shape[0]
    
    np.savez('train_data_re.npz', data_xy=data_xy, panel_data=panel_data, aoa=aoa, re=re, cl=cl, cd=cd, cm=cm)

else:
    data = np.load('train_data_re.npz')
    data_xy = data['data_xy']
    panel_data = data['panel_data']
    aoa = data['aoa']
    re = data['re']
    cl = data['cl']
    cd = data['cd']
    cm = data['cm']
    
#%% 
if im == 1:       
    xtrain, ytrain = training_data(data_xy,panel_data,aoa,re,cl,cd)
elif im == 2:       
    xtrain, ytrain = training_data_cl_cd(data_xy,panel_data,aoa,re,cl,cd)
elif im == 3:       
    xtrain, ytrain = training_data_cp(data_xy,panel_data,aoa,re,cl,cd)
elif im == 4:       
    xtrain, ytrain = training_data_cl(data_xy,panel_data,aoa,re,cl,cd)
    
#%%
sc_input = MinMaxScaler(feature_range=(-1,1))
sc_input = sc_input.fit(xtrain)
xtrain_sc = sc_input.transform(xtrain)

sc_output = MinMaxScaler(feature_range=(-1,1))
sc_output = sc_output.fit(ytrain)
ytrain_sc = sc_output.transform(ytrain)

xtrain, ytrain = xtrain_sc, ytrain_sc

xtrain1 = np.copy(xtrain[:,:2*num_xy])
xtrain2 = np.copy(xtrain[:,2*num_xy:])

#%%
training = True
nf1 = xtrain1.shape[1]
nf2 = xtrain2.shape[1]
nl = ytrain.shape[1]

#%%
for i in range(1,31):
    seed_number = int(i*10)
    print(seed_number)
    import random
    random.seed(seed_number)
    from numpy.random import seed
    seed(seed_number)
    from tensorflow import set_random_seed
    set_random_seed(seed_number)
    
    """## Define Neural Network Architecture"""
    n_layers = 2
    n_neurons = [20,20]
    lr = 0.001
    epochs = 800
    batch_size = 16
    
    x1 = Input(shape=(nf1,))
    x2 = Input(shape=(nf2,))
    
#    x = Dense(20,activation='relu',kernel_regularizer=regularizers.l2(1e-2))(x1)
#    x = Dense(20,activation='relu',kernel_regularizer=regularizers.l2(1e-2))(x)
    
    x = Dense(20,activation='relu')(x1)
    x = Dense(20,activation='relu')(x)
    
    x = concatenate(inputs=[x, x2])

#    x = Dense(20,activation='relu',kernel_regularizer=regularizers.l2(1e-2))(x)
#    x = Dense(20,activation='relu',kernel_regularizer=regularizers.l2(1e-2))(x)
    
    x = Dense(20,activation='relu')(x)
    x = Dense(20,activation='relu')(x)
    
    output = Dense(nl,activation='linear')(x)
    
    model = Model(inputs=[x1, x2], outputs=output)
    
    print(model.summary())
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    model.compile(loss='mean_squared_error', optimizer=opt)
    
    """## Train the model"""
    
    es = EarlyStopping(monitor='loss', patience=40)
    
    tb = TensorBoard(log_dir='./logs', histogram_freq=10, batch_size=batch_size,
                    write_graph=True, write_images=True, write_grads=True)
    
    history_callback = model.fit(x=[xtrain1, xtrain2], y=ytrain, batch_size=batch_size, epochs=epochs, \
             validation_split=0.2, shuffle=True, callbacks=[tb])
    
    loss = history_callback.history["loss"]
    val_loss = history_callback.history["val_loss"]
    
    plt.figure()
    epochs = range(1, len(loss) + 1)
    plt.semilogy(epochs, loss, 'b', label='Training loss')
    plt.semilogy(epochs, val_loss, 'r', label='Validationloss')
    plt.title('Training and validation loss')
    plt.legend()
    filename = f'./model_{im}/loss_{im}_{seed_number}.png'
    plt.savefig(filename, dpi = 300)
    plt.show()
    
    model.save(f'./model_{im}/ann_model_{im}_{seed_number}.h5')
    
    tf.reset_default_graph()
    
#%%
for i in range(1,31):
    seed_number = int(i*10)
    loaded_model = tf.keras.models.load_model(f'./model_{im}/ann_model_{im}_{seed_number}.h5')
        
    data_path = '../airfoil_shapes_re_v2/'   
    #data_path = '../airfoil_shapes/'   
    
    db = data_path + 'test_data.csv'
    df = pd.read_csv(db, encoding='utf-8')
    
    df.columns = col_name
    
    df = df[df['CL'].notna()]
    
    df = df[df["Airfoil"] == 'naca23012']
    df = df[df["RE"] == 3e6]
    
    num_samples = df.shape[0]
    
    panel_data_test = np.zeros((num_samples,num_cp+2))
    data_xy_test = np.zeros((num_samples,num_xy,2))
    
    airfoil_names_test = []
    aoa_test = np.zeros(num_samples)
    re_test = np.zeros(num_samples)
    cl_test = np.zeros(num_samples)
    cd_test = np.zeros(num_samples)
    cm_test = np.zeros(num_samples)
    
    airfoils_xcoords_test = df.iloc[:,1:202].values
    airfoils_ycoords_test = df.iloc[:,202:403].values
    
    counter = 0
    cd_scale_factor = 10.0
    
    generate_new_test_data = True
    
    if generate_new_test_data :
        for index, row in df.iterrows():
            airfoil_names_test.append(row.Airfoil)
            aoa_test[counter] = row.AOA2
            re_test[counter] = row.RE
            cl_test[counter] = row.CL
            cd_test[counter] = row.CD*cd_scale_factor
            cm_test[counter] = row.CM
            
            data_xy_test[counter,:,0] = airfoils_xcoords_test[counter,:]
            data_xy_test[counter,:,1] = airfoils_ycoords_test[counter,:]
            
            CL, CDP, Cp, pp = panel(data_xy_test[counter,:,:], alfader=row.AOA2)
            panel_data_test[counter,0], panel_data_test[counter,1], panel_data_test[counter,2:] = CL,CDP,Cp
            
            print(counter, row.Airfoil, row.AOA2)
            counter +=1    
        
        np.savez('test_data_re_23012.npz',airfoil_names_test=airfoil_names_test, data_xy_test=data_xy_test,panel_data_test=panel_data_test,aoa_test=aoa_test,re_test=re_test,cl_test=cl_test,cd_test=cd_test,cm_test=cm_test)
    
    else:
    #    data = np.load('test_data.npz')
        data = np.load('test_data_re_23012.npz')
        airfoil_names_test = data['airfoil_names_test']
        data_xy_test = data['data_xy_test']
        panel_data_test = data['panel_data_test']
        aoa_test = data['aoa_test']
        re_test = data['re_test']
        cl_test = data['cl_test']
        cd_test = data['cd_test']
        cm_test = data['cm_test']
    
    if im == 1:       
        xtest, ytest = training_data(data_xy_test,panel_data_test,aoa_test,re_test,cl_test,cd_test)
    elif im == 2:      
        xtest, ytest = training_data_cl_cd(data_xy_test,panel_data_test,aoa_test,re_test,cl_test,cd_test)
    elif im == 3:       
        xtest, ytest = training_data_cp(data_xy_test,panel_data_test,aoa_test,re_test,cl_test,cd_test)
    elif im == 4:      
        xtest, ytest = training_data_cl(data_xy_test,panel_data_test,aoa_test,re_test,cl_test,cd_test)
    
    xtest_sc = sc_input.transform(xtest[:,:])  
    xtest_sc1 = np.copy(xtest_sc[:,:2*num_xy])
    xtest_sc2 = np.copy(xtest_sc[:,2*num_xy:])
    
    ypred_sc = loaded_model.predict([xtest_sc1,xtest_sc2]) 
    ypred_sc1 = loaded_model.predict([xtest_sc1,xtest_sc2]) 
    ypred = sc_output.inverse_transform(ypred_sc)
    
    fig,ax = plt.subplots(1,2, figsize=(12,5))
    
    ax[0].plot(aoa_test[:num_samples], ytest[:num_samples,0], 'ro-', label='Xfoil NACA23012')
    ax[0].plot(aoa_test[:num_samples], ypred[:num_samples,0], 'bo-', label=f'ML-{im} NACA23012')
    #ax[0].plot(a, cl, 'bo-', label='Xfoil NACA0012')
    ax[0].legend()
    
    ax[1].plot(aoa_test[:num_samples], ytest[:num_samples,0], 'ro-', label='Xfoil NACA0012 ')
    ax[1].plot(aoa_test[:num_samples], ypred[:num_samples,0], 'bo-', label=f'ML-{im} NACA0012 ')
    #ax[1].plot(a, cd, 'bo-', label='Xfoil NACA0012 ')
    ax[1].legend()
    
    ax[0].set_ylabel(r'$C_L$')
    ax[0].set_xlabel(r'$\alpha$')
    
    ax[1].set_ylabel(r'$C_D$')
    ax[1].set_xlabel(r'$\alpha$')
    
    fig.tight_layout()
    plt.show()
    fig.savefig(f'./model_{im}/result_{im}_{seed_number}_23012.png', dpi=300)
    
    np.savez(f'./model_{im}/result_{im}_{seed_number}_23012.npz', xtest = xtest, ytest=ytest, ypred=ypred)
    
    file = open(f"./model_{im}/output_{im}_{seed_number}_23012.txt", "w", buffering=1)
    
    for i in range(num_samples):
        print(airfoil_names_test[i] + f' CL_test = %.4f, CL_pred = %.4f'%(ytest[i,0], ypred[i,0]))
        print(airfoil_names_test[i] + f' CD_test = %.4f, CD_pred = %.4f'%(ytest[i,0], ypred[i,0]))
        
        print(airfoil_names_test[i] + f' CL_test = %.4f, CL_pred = %.4f'%(ytest[i,0], ypred[i,0]), file=file)
        print(airfoil_names_test[i] + f' CD_test = %.4f, CD_pred = %.4f'%(ytest[i,0], ypred[i,0]), file=file)
    
    file.close()

#%%
for i in range(1,31):
    seed_number = int(i*10)
    loaded_model = tf.keras.models.load_model(f'./model_{im}/ann_model_{im}_{seed_number}.h5')
        
    data_path = '../airfoil_shapes_re_v2/'   
    #data_path = '../airfoil_shapes/'   
    
    db = data_path + 'test_data.csv'
    df = pd.read_csv(db, encoding='utf-8')
    
    df.columns = col_name
    
    df = df[df['CL'].notna()]
    
    df = df[df["Airfoil"] == 'naca23024']
    df = df[df["RE"] == 3e6]
    
    num_samples = df.shape[0]
    
    panel_data_test = np.zeros((num_samples,num_cp+2))
    data_xy_test = np.zeros((num_samples,num_xy,2))
    
    airfoil_names_test = []
    aoa_test = np.zeros(num_samples)
    re_test = np.zeros(num_samples)
    cl_test = np.zeros(num_samples)
    cd_test = np.zeros(num_samples)
    cm_test = np.zeros(num_samples)
    
    airfoils_xcoords_test = df.iloc[:,1:202].values
    airfoils_ycoords_test = df.iloc[:,202:403].values
    
    counter = 0
    cd_scale_factor = 10.0
    
    generate_new_test_data = True
    
    if generate_new_test_data :
        for index, row in df.iterrows():
            airfoil_names_test.append(row.Airfoil)
            aoa_test[counter] = row.AOA2
            re_test[counter] = row.RE
            cl_test[counter] = row.CL
            cd_test[counter] = row.CD*cd_scale_factor
            cm_test[counter] = row.CM
            
            data_xy_test[counter,:,0] = airfoils_xcoords_test[counter,:]
            data_xy_test[counter,:,1] = airfoils_ycoords_test[counter,:]
            
            CL, CDP, Cp, pp = panel(data_xy_test[counter,:,:], alfader=row.AOA2)
            panel_data_test[counter,0], panel_data_test[counter,1], panel_data_test[counter,2:] = CL,CDP,Cp
            
            print(counter, row.Airfoil, row.AOA2)
            counter +=1    
        
        np.savez('test_data_re_23024.npz',airfoil_names_test=airfoil_names_test, data_xy_test=data_xy_test,panel_data_test=panel_data_test,aoa_test=aoa_test,re_test=re_test,cl_test=cl_test,cd_test=cd_test,cm_test=cm_test)
    
    else:
    #    data = np.load('test_data.npz')
        data = np.load('test_data_re_23024.npz')
        airfoil_names_test = data['airfoil_names_test']
        data_xy_test = data['data_xy_test']
        panel_data_test = data['panel_data_test']
        aoa_test = data['aoa_test']
        re_test = data['re_test']
        cl_test = data['cl_test']
        cd_test = data['cd_test']
        cm_test = data['cm_test']
    
    if im == 1:       
        xtest, ytest = training_data(data_xy_test,panel_data_test,aoa_test,re_test,cl_test,cd_test)
    elif im == 2:      
        xtest, ytest = training_data_cl_cd(data_xy_test,panel_data_test,aoa_test,re_test,cl_test,cd_test)
    elif im == 3:       
        xtest, ytest = training_data_cp(data_xy_test,panel_data_test,aoa_test,re_test,cl_test,cd_test)
    elif im == 4:      
        xtest, ytest = training_data_cl(data_xy_test,panel_data_test,aoa_test,re_test,cl_test,cd_test)
    
    xtest_sc = sc_input.transform(xtest[:,:])  
    xtest_sc1 = np.copy(xtest_sc[:,:2*num_xy])
    xtest_sc2 = np.copy(xtest_sc[:,2*num_xy:])
    
    ypred_sc = loaded_model.predict([xtest_sc1,xtest_sc2]) 
    ypred_sc1 = loaded_model.predict([xtest_sc1,xtest_sc2]) 
    ypred = sc_output.inverse_transform(ypred_sc)
    
    fig,ax = plt.subplots(1,2, figsize=(12,5))
    
    ax[0].plot(aoa_test[:num_samples], ytest[:num_samples,0], 'ro-', label='Xfoil NACA23024')
    ax[0].plot(aoa_test[:num_samples], ypred[:num_samples,0], 'bo-', label=f'ML-{im} NACA23024')
    #ax[0].plot(a, cl, 'bo-', label='Xfoil NACA0012')
    ax[0].legend()
    
    ax[1].plot(aoa_test[:num_samples], ytest[:num_samples,0], 'ro-', label='Xfoil NACA0024')
    ax[1].plot(aoa_test[:num_samples], ypred[:num_samples,0], 'bo-', label=f'ML-{im} NACA0024')
    #ax[1].plot(a, cd, 'bo-', label='Xfoil NACA0012 ')
    ax[1].legend()
    
    ax[0].set_ylabel(r'$C_L$')
    ax[0].set_xlabel(r'$\alpha$')
    
    ax[1].set_ylabel(r'$C_D$')
    ax[1].set_xlabel(r'$\alpha$')
    
    fig.tight_layout()
    plt.show()
    fig.savefig(f'./model_{im}/result_{im}_{seed_number}_23024.png', dpi=300)
    
    np.savez(f'./model_{im}/result_{im}_{seed_number}_23024.npz', xtest = xtest, ytest=ytest, ypred=ypred)
    
    file = open(f"./model_{im}/output_{im}_{seed_number}_23024.txt", "w", buffering=1)
    
    for i in range(num_samples):
        print(airfoil_names_test[i] + f' CL_test = %.4f, CL_pred = %.4f'%(ytest[i,0], ypred[i,0]))
        print(airfoil_names_test[i] + f' CD_test = %.4f, CD_pred = %.4f'%(ytest[i,0], ypred[i,0]))
        
        print(airfoil_names_test[i] + f' CL_test = %.4f, CL_pred = %.4f'%(ytest[i,0], ypred[i,0]), file=file)
        print(airfoil_names_test[i] + f' CD_test = %.4f, CD_pred = %.4f'%(ytest[i,0], ypred[i,0]), file=file)
    
    file.close()
    