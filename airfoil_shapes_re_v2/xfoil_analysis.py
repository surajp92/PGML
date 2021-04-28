#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:47:31 2020

@author: suraj
"""
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from xfoil import XFoil
from xfoil.model import Airfoil

#%%
import glob
import os

#os.chdir(r'directory where the files are located')
myFiles = glob.glob('*.txt')
print(myFiles)

#%%
xf = XFoil()

from xfoil.test import naca0012
xf.airfoil = naca0012

coords = naca0012.coords

xf.Re = 1e6
xf.max_iter = 80
a, cl, cd, cm, cpmin = xf.aseq(-20, 22, 2)

#%%
Re_list = [1e6,2e6,3e6,4e6]
A_list = [0,2,4,6]
B_list = [0,2,4,6]
C_list = [0,1]
D_list = [2,4,6]


if os.path.isfile('train_data_naca4.csv'):
    os.remove('train_data_naca4.csv')
  
for A in A_list:
        for B in B_list:
            for C in C_list:
                for D in D_list:
                    for re in Re_list:
                        file = f'naca{A}{B}{C}{D}'
                        filename = f'{file}.txt'
                        
                        airfoil_xy = np.loadtxt(filename, skiprows=0)
                        airfoil_xy = np.flip(airfoil_xy, axis=0)
                        airfoil_xy = np.round(airfoil_xy,6)
                        
                        xfc = XFoil()
                        xfc.airfoil = Airfoil(airfoil_xy[:,0], airfoil_xy[:,1])
                        
                        xfc.Re = re
                        xfc.max_iter = 40
                        ac, clc, cdc, cmc, cpminc = xfc.aseq(-20, 22, 2)
                        
                        angle_counts = ac.shape[0]
                        list_results = []
                        list_result = []
                        
                        for i in range(angle_counts):
                            list_result.append(file)
                            list_result.extend(airfoil_xy[:,0])
                            list_result.extend(airfoil_xy[:,1])
                            list_result.append(a[i])
                            list_result.append(ac[i])
                            list_result.append(re)
                            list_result.append(clc[i])
                            list_result.append(cdc[i])
                            list_result.append(cmc[i])
                            list_result.append(cpminc[i])
                            
                            list_results.append(list_result)
                            list_result = []
                            
                        for item in list_results:
                            with open('train_data_naca4.csv', 'a') as csvfile:
                                writer = csv.writer(csvfile, delimiter=",")
                                writer.writerow(item) 

#%%
if os.path.isfile('train_data_naca210.csv'):
    os.remove('train_data_naca210.csv')

for C in C_list:
    for D in D_list:
        for re in Re_list:
            file = f'naca210{C}{D}'      
            filename = f'{file}.txt'
                        
            airfoil_xy = np.loadtxt(filename, skiprows=0)
            airfoil_xy = np.flip(airfoil_xy, axis=0)
            airfoil_xy = np.round(airfoil_xy,6)

            xfc = XFoil()
            xfc.airfoil = Airfoil(airfoil_xy[:,0], airfoil_xy[:,1])
            
            xfc.Re = re
            xfc.max_iter = 40
            ac, clc, cdc, cmc, cpminc = xfc.aseq(-20, 22, 2)
            
            angle_counts = ac.shape[0]
            list_results = []
            list_result = []
            
            for i in range(angle_counts):
                list_result.append(file)
                list_result.extend(airfoil_xy[:,0].flatten())
                list_result.extend(airfoil_xy[:,1])
                list_result.append(a[i])
                list_result.append(ac[i])
                list_result.append(re)
                list_result.append(clc[i])
                list_result.append(cdc[i])
                list_result.append(cmc[i])
                list_result.append(cpminc[i])
                
                list_results.append(list_result)
                list_result = []
                
            for item in list_results:
                with open('train_data_naca210.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=",")
                    writer.writerow(item)  
                    
#%%
if os.path.isfile('train_data_naca220.csv'):
    os.remove('train_data_naca220.csv')

for C in C_list:
    for D in D_list:
        for re in Re_list:
            file = f'naca220{C}{D}'      
            filename = f'{file}.txt'
                        
            airfoil_xy = np.loadtxt(filename, skiprows=0)
            airfoil_xy = np.flip(airfoil_xy, axis=0)
            airfoil_xy = np.round(airfoil_xy,6)
            
            xfc = XFoil()
            xfc.airfoil = Airfoil(airfoil_xy[:,0], airfoil_xy[:,1])
            
            xfc.Re = re
            xfc.max_iter = 40
            ac, clc, cdc, cmc, cpminc = xfc.aseq(-20, 22, 2)
            
            angle_counts = ac.shape[0]
            list_results = []
            list_result = []
            
            for i in range(angle_counts):
                list_result.append(file)
                list_result.extend(airfoil_xy[:,0].flatten())
                list_result.extend(airfoil_xy[:,1])
                list_result.append(a[i])
                list_result.append(ac[i])
                list_result.append(re)
                list_result.append(clc[i])
                list_result.append(cdc[i])
                list_result.append(cmc[i])
                list_result.append(cpminc[i])
                
                list_results.append(list_result)
                list_result = []
                
            for item in list_results:
                with open('train_data_naca220.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=",")
                    writer.writerow(item)    

#%%
if os.path.isfile('train_data_naca230.csv'):
    os.remove('train_data_naca230.csv')

for C in C_list:
    for D in D_list:
        for re in Re_list:
            file = f'naca230{C}{D}'      
            filename = f'{file}.txt'
                        
            airfoil_xy = np.loadtxt(filename, skiprows=0)
            airfoil_xy = np.flip(airfoil_xy, axis=0)
            airfoil_xy = np.round(airfoil_xy,6)
            
            xfc = XFoil()
            xfc.airfoil = Airfoil(airfoil_xy[:,0], airfoil_xy[:,1])
            
            xfc.Re = re
            xfc.max_iter = 40
            ac, clc, cdc, cmc, cpminc = xfc.aseq(-20, 22, 2)
            
            angle_counts = ac.shape[0]
            list_results = []
            list_result = []
            
            for i in range(angle_counts):
                list_result.append(file)
                list_result.extend(airfoil_xy[:,0].flatten())
                list_result.extend(airfoil_xy[:,1])
                list_result.append(a[i])
                list_result.append(ac[i])
                list_result.append(re)
                list_result.append(clc[i])
                list_result.append(cdc[i])
                list_result.append(cmc[i])
                list_result.append(cpminc[i])
                
                list_results.append(list_result)
                list_result = []
                
            for item in list_results:
                with open('train_data_naca230.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=",")
                    writer.writerow(item)   

#%%
if os.path.isfile('train_data_naca240.csv'):
    os.remove('train_data_naca240.csv')

for C in C_list:
    for D in D_list:
        for re in Re_list:
            file = f'naca240{C}{D}'      
            filename = f'{file}.txt'
                        
            airfoil_xy = np.loadtxt(filename, skiprows=0)
            airfoil_xy = np.flip(airfoil_xy, axis=0)
            airfoil_xy = np.round(airfoil_xy,6)
       
            xfc = XFoil()
            xfc.airfoil = Airfoil(airfoil_xy[:,0], airfoil_xy[:,1])
            
            xfc.Re = re
            xfc.max_iter = 40
            ac, clc, cdc, cmc, cpminc = xfc.aseq(-20, 22, 2)
            
            angle_counts = ac.shape[0]
            list_results = []
            list_result = []
            
            for i in range(angle_counts):
                list_result.append(file)
                list_result.extend(airfoil_xy[:,0].flatten())
                list_result.extend(airfoil_xy[:,1])
                list_result.append(a[i])
                list_result.append(ac[i])
                list_result.append(re)
                list_result.append(clc[i])
                list_result.append(cdc[i])
                list_result.append(cmc[i])
                list_result.append(cpminc[i])
                
                list_results.append(list_result)
                list_result = []
                
            for item in list_results:
                with open('train_data_naca240.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=",")
                    writer.writerow(item)   

#%%
if os.path.isfile('train_data_naca250.csv'):
    os.remove('train_data_naca250.csv')

for C in C_list:
    for D in D_list:
        for re in Re_list:
            file = f'naca250{C}{D}'      
            filename = f'{file}.txt'
                        
            airfoil_xy = np.loadtxt(filename, skiprows=0)
            airfoil_xy = np.flip(airfoil_xy, axis=0)
            airfoil_xy = np.round(airfoil_xy,6)
 
            xfc = XFoil()
            xfc.airfoil = Airfoil(airfoil_xy[:,0], airfoil_xy[:,1])
            
            xfc.Re = re
            xfc.max_iter = 40
            ac, clc, cdc, cmc, cpminc = xfc.aseq(-20, 22, 2)
            
            angle_counts = ac.shape[0]
            list_results = []
            list_result = []
            
            for i in range(angle_counts):
                list_result.append(file)
                list_result.extend(airfoil_xy[:,0].flatten())
                list_result.extend(airfoil_xy[:,1])
                list_result.append(a[i])
                list_result.append(ac[i])
                list_result.append(re)
                list_result.append(clc[i])
                list_result.append(cdc[i])
                list_result.append(cmc[i])
                list_result.append(cpminc[i])
                
                list_results.append(list_result)
                list_result = []
                
            for item in list_results:
                with open('train_data_naca250.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=",")
                    writer.writerow(item)   
                  
#%%
fig,ax = plt.subplots(1,2, figsize=(12,5))

ax[0].plot(ac, clc, 'ro-', label='Xfoil NACA0012 (N = 201)')
#ax[0].plot(a, cl, 'bo-', label='Xfoil NACA0012')
ax[0].legend()

ax[1].plot(ac, cdc, 'ro-', label='Xfoil NACA0012 (N = 201)')
#ax[1].plot(a, cd, 'bo-', label='Xfoil NACA0012 ')
ax[1].legend()

plt.show()
fig.tight_layout()
#fig.savefig('naca0012_aoa.png', dpi=300)