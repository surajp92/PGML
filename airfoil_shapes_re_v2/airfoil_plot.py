#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 15:59:07 2020

@author: suraj
"""
import numpy as np
import matplotlib.pyplot as plt

fig1,ax1 = plt.subplots(1,1, figsize=(12,10))

ax1.set_xlim([-0.1,1.1])
ax1.set_ylim([-0.5,0.5])
        
C_list = [0]
D_list = [6,8]  
for C in C_list:
    for D in D_list:
        A = 0
        B = 0
        file = f'naca{A}{B}{C}{D}'
        filename = f'{file}.txt'
        
        airfoil_xy = np.loadtxt(filename, skiprows=0)
        
        ax1.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'k', lw=1, alpha=0.2)
        
        fig,ax = plt.subplots(1,1, figsize=(12,10))
        
        ax.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'k', lw=4)
    #                    ax.fill(airfoil_xy[:,0], airfoil_xy[:,1], 'k') 
        
        ax.set_xlim([-0.1,1.1])
        ax.set_ylim([-0.5,0.5])
        
    #                    ax.set_xticks([])
    #                    ax.set_yticks([])
        
        fig.tight_layout()
        plt.show()
        fig.savefig(f'./png/{file}.png',dpi=100)

C_list = [1]
D_list = [0,2,4,6] 
for C in C_list:
    for D in D_list:
        A = 0
        B = 0
        file = f'naca{A}{B}{C}{D}'
        filename = f'{file}.txt'
        
        airfoil_xy = np.loadtxt(filename, skiprows=0)
        
        ax1.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'k', lw=1, alpha=0.2)
        
        fig,ax = plt.subplots(1,1, figsize=(12,10))
        
        ax.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'k', lw=4)
    #                    ax.fill(airfoil_xy[:,0], airfoil_xy[:,1], 'k') 
        
        ax.set_xlim([-0.1,1.1])
        ax.set_ylim([-0.5,0.5])
        
    #                    ax.set_xticks([])
    #                    ax.set_yticks([])
        
        fig.tight_layout()
        plt.show()
        fig.savefig(f'./png/{file}.png',dpi=100)
                    
A_list = [2,3,4,5,6]
B_list = [2,3,4,5,6]
C_list = [0]
D_list = [6,8]

for A in A_list:
        for B in B_list:
            for C in C_list:
                for D in D_list:
                    file = f'naca{A}{B}{C}{D}'
                    filename = f'{file}.txt'
                    
                    airfoil_xy = np.loadtxt(filename, skiprows=0)
                    
                    ax1.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'k', lw=1, alpha=0.2)
                    
                    fig,ax = plt.subplots(1,1, figsize=(12,10))
                    
                    ax.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'k', lw=4)
#                    ax.fill(airfoil_xy[:,0], airfoil_xy[:,1], 'k') 
                    
                    ax.set_xlim([-0.1,1.1])
                    ax.set_ylim([-0.5,0.5])
                    
#                    ax.set_xticks([])
#                    ax.set_yticks([])
                    
                    fig.tight_layout()
                    plt.show()
                    fig.savefig(f'./png/{file}.png',dpi=100)

C_list = [1]
D_list = [0,2,4,6]
for A in A_list:
        for B in B_list:
            for C in C_list:
                for D in D_list:
                    file = f'naca{A}{B}{C}{D}'
                    filename = f'{file}.txt'
                    
                    airfoil_xy = np.loadtxt(filename, skiprows=0)
                    
                    ax1.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'k', lw=1, alpha=0.2)
                    
                    fig,ax = plt.subplots(1,1, figsize=(12,10))
                    
                    ax.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'k', lw=4)
#                    ax.fill(airfoil_xy[:,0], airfoil_xy[:,1], 'k') 
                    
                    ax.set_xlim([-0.1,1.1])
                    ax.set_ylim([-0.5,0.5])
                    
#                    ax.set_xticks([])
#                    ax.set_yticks([])
                    
                    fig.tight_layout()
                    plt.show()
                    fig.savefig(f'./png/{file}.png',dpi=100)
                    
#%%
for C in C_list:
    for D in D_list:
        file = f'naca210{C}{D}'      
        filename = f'{file}.txt'                    
        
        airfoil_xy = np.loadtxt(filename, skiprows=0)
        
        ax1.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'k', lw=1, alpha=0.2)
                
        fig,ax = plt.subplots(1,1, figsize=(12,10))
        
        ax.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'k', lw=4)
#        ax.fill(airfoil_xy[:,0], airfoil_xy[:,1], 'k') 
        
        ax.set_xlim([-0.1,1.1])
        ax.set_ylim([-0.5,0.5])
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        fig.tight_layout()
        plt.show()
        fig.savefig(f'./png/{file}.png',dpi=100)

#%%
for C in C_list:
    for D in D_list:
        file = f'naca220{C}{D}'      
        filename = f'{file}.txt'                    
        
        airfoil_xy = np.loadtxt(filename, skiprows=0)
        
        ax1.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'k', lw=1, alpha=0.2)
        
        fig,ax = plt.subplots(1,1, figsize=(12,10))
        
        ax.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'k', lw=4)
#        ax.fill(airfoil_xy[:,0], airfoil_xy[:,1], 'k') 
        
        ax.set_xlim([-0.1,1.1])
        ax.set_ylim([-0.5,0.5])
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        fig.tight_layout()
        plt.show()
        fig.savefig(f'./png/{file}.png',dpi=100)    

#%%
for C in C_list:
    for D in D_list:
        file = f'naca230{C}{D}'      
        filename = f'{file}.txt'                    
        
        airfoil_xy = np.loadtxt(filename, skiprows=0)
               
        ax1.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'k', lw=1, alpha=0.2)
        
        fig,ax = plt.subplots(1,1, figsize=(12,10))
        
        ax.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'k', lw=4)
#        ax.fill(airfoil_xy[:,0], airfoil_xy[:,1], 'k') 
        
        ax.set_xlim([-0.1,1.1])
        ax.set_ylim([-0.5,0.5])
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        fig.tight_layout()
        plt.show()
        fig.savefig(f'./png/{file}.png',dpi=100)    

#%%
for C in C_list:
    for D in D_list:
        file = f'naca240{C}{D}'      
        filename = f'{file}.txt'                    
        
        airfoil_xy = np.loadtxt(filename, skiprows=0)
          
        ax1.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'k', lw=1, alpha=0.2)
        
        fig,ax = plt.subplots(1,1, figsize=(12,10))
        
        ax.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'k', lw=4)
#        ax.fill(airfoil_xy[:,0], airfoil_xy[:,1], 'k') 
        
        ax.set_xlim([-0.1,1.1])
        ax.set_ylim([-0.5,0.5])
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        fig.tight_layout()
        plt.show()
        fig.savefig(f'./png/{file}.png',dpi=100)        

#%%
for C in C_list:
    for D in D_list:
        file = f'naca250{C}{D}'      
        filename = f'{file}.txt'                    
        
        airfoil_xy = np.loadtxt(filename, skiprows=0)
        
        ax1.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'k', lw=1, alpha=0.2)
        
        fig,ax = plt.subplots(1,1, figsize=(12,10))
        
        ax.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'k', lw=4)
#        ax.fill(airfoil_xy[:,0], airfoil_xy[:,1], 'k') 
        
        ax.set_xlim([-0.1,1.1])
        ax.set_ylim([-0.5,0.5])
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        fig.tight_layout()
        plt.show()
        fig.savefig(f'./png/{file}.png',dpi=100)        

#%%
file = f'naca23012'
filename = f'{file}.txt'                           
airfoil_xy = np.loadtxt(filename, skiprows=0)
ax1.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'b', lw=2, alpha=0.8)

file = f'naca23024'
filename = f'{file}.txt'                           
airfoil_xy = np.loadtxt(filename, skiprows=0)
ax1.plot(airfoil_xy[:,0], airfoil_xy[:,1], 'r', lw=2, alpha=0.8)

fig1.tight_layout()
plt.show()
fig1.savefig(f'./png/all_airfoils.png',dpi=100)          