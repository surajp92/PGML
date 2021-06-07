#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:51:15 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

#%%
filename = f'./Similarity_Solution/su2.csv'
data_blasisus = np.genfromtxt(filename, delimiter=',')

#filename = f'./OpenFoam_Laminar/laminarFlatPlate/postProcessing/singleGraph/1/U.csv'
#U0 = 1.0
#nu = 1.0e-06
#x = 1.5

filename = f'./OpenFoam_Laminar/laminarFlatPlate/postProcessing/singleGraph/1/y_ux.csv'
U0 = 69.4
nu = 1.388e-05
x = 1.5

#f = open(filename,'r') 
#y,ux,uy,uz = zip(*[l.split() for l in f])
data_cfd = np.genfromtxt(filename, delimiter=',')

y = data_cfd[:,0]*np.sqrt(U0/(nu*x))
ux = data_cfd[:,1]/U0

#%%
fig, ax = plt.subplots(1,1,figsize=(6,5))
ax.plot(data_blasisus[:,0], data_blasisus[:,1], ls='None', 
        marker='o', color='r', ms=8, fillstyle='none')
ax.plot(ux, y, 'ks-', lw=2)
#ax.plot(data_cfd[:,1], data_cfd[:,0], 'k-', lw=2)

ax.grid()
ax.set_xlim([0,1.1])
ax.set_ylim([0,50])
plt.show()