#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 13:45:40 2021

@author: suraj
"""

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

font = {'size'   : 14}    
plt.rc('font', **font)

filename = f'../../../Similarity_Solution/su2.csv'
data_blasisus = np.genfromtxt(filename, delimiter=',')

## grid is the central object in VTK where every field is added on to grid

grid_5 = pv.UnstructuredGrid(f'./Re_0.5e+5/VTK/Re_0.5e+5_10000.vtk')
grid_10 = pv.UnstructuredGrid(f'./Re_1.0e+5/VTK/Re_1.0e+5_10000.vtk')

grid = [grid_5, grid_10]
u_list = [0.5, 1.0]

fig, ax = plt.subplots(2,4,figsize=(15,10),sharey=True,)

non_dim = False

x_list = [0.005,0.015,0.025,0.05]
#for x in [0.005,0.015,0.025,0.03]:
for j in range(4):    
    x = x_list[j]
    ax[0,j].set_title(f'$x={x}$')
    
    a = [x,0.0,0.0005]
    b = [x,0.005,0.0005]
    line = pv.Line(a, b, resolution=200)
    
    for k in range(2):    
        result = grid[k].probe(line)
       
        y = np.array(line.points)[:,1]
        ux = result['U'][:,0]
        uy = result['U'][:,1]*100
        
        U0 = u_list[k]
        nu = 1.0e-06
        
        if non_dim:
            y = y*np.sqrt(U0/(nu*x))
        else:
            y = y
            
        ux = ux/U0
        
        if k == 0:
            ax[0,j].plot(ux, y, '-', lw=2, label=f'Re = {U0}E5')
            ax[1,j].plot(uy, y, '-', lw=2, label=f'Re = {U0}E5')
        else:
            ax[0,j].plot(ux, y, '-.', lw=2, label=f'Re = {U0}E5')
            ax[1,j].plot(uy, y, '-.', lw=2, label=f'Re = {U0}E5')

for j in range(4): 
    
    if non_dim:
        ax[0,j].plot(data_blasisus[:,0], data_blasisus[:,1], ls='none', 
           marker='o', color='k', ms=8, fillstyle='none', label='Blasius')
        ax[0,j].set_ylim([0,6])
    else:
        ax[0,j].set_ylim([0,0.002])
        
    ax[0,j].legend()
    ax[0,j].grid()
    ax[0,j].set_xlim([0,1.1])
            
    ax[1,j].legend()
    ax[1,j].grid()
    ax[1,j].set_xlim([0,0.015*100])

    ax[0,j].set_xlabel('$u/u_\infty$')
    ax[1,j].set_xlabel('$v/u_\infty (1 X 10^{-2})$')
    
#    ax[j,1].set_ylim([0,0.002])
#    ax[j,1].set_ylabel('$\eta$')
plt.show()
if non_dim:
    ax[0,0].set_ylabel('$\eta$')
    ax[1,0].set_ylabel('$\eta$')
    fig.savefig('non_dim_vel_profile.png', dpi=300, bbox_inches='tight')
else:    
    ax[0,0].set_ylabel('$y$')
    ax[1,0].set_ylabel('$y$')
    fig.savefig('dim_vel_profile.png', dpi=300, bbox_inches='tight')

