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

filename = f'./OpenFoam_Laminar/laminarFlatPlate/pinn_paper_mesh/postProcessing/singleGraph/1/U.csv'
U0 = 1.0
nu = 1.0e-06
x = 0.03

#filename = f'./OpenFoam_Laminar/laminarFlatPlate/postProcessing/singleGraph/1/y_ux.csv'
#U0 = 69.4
#nu = 1.388e-05
#x = 1.5

#f = open(filename,'r') 
#y,ux,uy,uz = zip(*[l.split() for l in f])
data_cfd = np.genfromtxt(filename, delimiter=',')

y = data_cfd[:,0]*np.sqrt(U0/(nu*x))
ux = data_cfd[:,1]/U0
uy = data_cfd[:,2]/U0

#%%
fig, ax = plt.subplots(1,2,figsize=(10,5))
ax[0].plot(data_blasisus[:,0], data_blasisus[:,1], ls='none', 
        marker='o', color='r', ms=8, fillstyle='none', label='Blasius')
ax[0].plot(ux, y, 'k-', lw=2, label='CFD-$u$')
#ax.plot(data_cfd[:,1], data_cfd[:,0], 'k-', lw=2)
ax[0].legend()
ax[0].grid()
ax[0].set_xlim([0,1.1])
ax[0].set_ylim([0,6])

ax[1].plot(uy, y, 'k-', lw=2, label='CFD-$v$')
ax[1].legend()
ax[1].grid()
ax[1].set_ylim([0,5])
plt.show()