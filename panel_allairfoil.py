#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 09:55:48 2020

@author: suraj
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#%%
def gauss(n1,D,E_):
    X = np.zeros(n1)
    for k in range(n1-1):
        for i in range(k+1,n1):
            rate = D[i,k]/D[k,k]
    #            print(rate)
            D[i,k] = D[i,k] - rate*D[k,k]
            E_[i] = E_[i] - rate*E_[k]
            
            for j in range(k+1,n1):
                D[i,j] = D[i,j] - rate*D[k,j]
    
    #    E_[-1] = 6.29755
    X[n1-1] = E_[n1-1]/D[n1-1,n1-1]
    
    for i in range(n1-2,-1,-1):
        X[i] = E_[i]
        for j in range(n1-1,i,-1):
    #            print(j)
            X[i] = X[i] - D[i,j]*X[j]
        X[i] = X[i]/D[i,i]
    
    return X
    
def panel(airfoil_xy, alfader=0):
    airfoil_xy = np.flip(airfoil_xy, axis=0)
    
#    airfoil_xy[0,1] = 0.0
#    airfoil_xy[-1,1] = 0.0
    
    n1 = airfoil_xy.shape[0]
    x1 = airfoil_xy[:,0]
    y1 = airfoil_xy[:,1]
    
    n = n1 - 1 # number of panel points
    
    alfa = alfader*np.pi/180 # angle of attack in radians
    U = 1.0 # free stream velocity
    
    # cirfoil cordinates as complex numbers
    z1 = np.zeros(n1, dtype=complex)
    z1 = x1 + 1j*y1


    # panel lengths
    dx = np.zeros(n)
    for i in range(n):
        dx[i] = np.abs(z1[i+1] - z1[i])
    #dx = np.abs(z1[1:] - z1[:-1])
    
    # panel angles  
    cs = np.real(z1[1:] - z1[:-1])/dx
    sn = np.imag(z1[1:] - z1[:-1])/dx
    
    t1 = cs + 1j*sn
    tx1 = np.conj(t1)
    
    # mid-point of panel coordinates
    pp = (z1[1:] + z1[:-1])/2.0
    
    #%%
    c = np.zeros((n,n), dtype=complex)
    aa = np.zeros((n,n))
    bb = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                c[i,j] = 0.0 - 0.5j
            else:
                c[i,j] = t1[i]*tx1[j]/(2.0*np.pi) * np.log((pp[i] - z1[j])/(pp[i] - z1[j+1]))
            
            aa[i,j] = np.real(c[i,j])
            bb[i,j] = np.imag(c[i,j])
    
    #%%
    toplambb = np.sum(bb, axis=1) 
    toplamaa = np.sum(aa, axis=1)        
    
    topb1 = np.sum(bb[0,:])
    topbn = np.sum(bb[n-1,:])
    
    #%%
    D = np.zeros((n1,n1))
    E = np.zeros(n1)
    X = np.zeros(n1)
    
    # Coefficient matrix
    for i in range(n):
        for j in range(n):
            D[i,j] = bb[i,j]
            D[i,n] = toplamaa[i]
            D[n,j] = aa[0,j] + aa[n-1,j]
            D[n,n] = -(topb1 + topbn)
    
    wxs = U*np.cos(alfa) - 1j*U*np.sin(alfa)
    
    E[:-1] = -np.imag(wxs*t1)
    E[n] = -np.real(wxs*(t1[0] + t1[n-1]))
           
    #%%    
    E = np.reshape(E,[-1,1])
    X = np.linalg.inv(D) @ E 
    X = np.reshape(X,[-1])
    
    #%%
#    X = gauss(n1,D,E)
    
    #%%
    aratop = aa @ X[:-1]        
        
    #%%
    Vt = np.real(wxs*t1) + aratop - toplambb*X[n]    
       
    #%%
    # compute pressure coefficient
    Cp = 1.0 - (Vt/U)**2
    
    #%%
    cxx = Cp*np.imag(z1[1:] - z1[:-1])
    cyy = -Cp*np.real(z1[1:] - z1[:-1])
    
    CX = np.sum(cxx)
    CY = np.sum(cyy)
    
    CL = CY*np.cos(alfa) - CX*np.sin(alfa)
    CDP = CY*np.sin(alfa) + CX*np.cos(alfa)
    
    return CL, CDP, Cp, pp
    

#%%
if __name__ == "__main__":
    filename = 'naca0012.txt'
    airfoil_xy = np.loadtxt(filename, skiprows=0)
    
    CL, CDP, Cp, pp = panel(airfoil_xy, alfader=0)
    
    print(f'CL = {np.round(CL,6)}')
    print(f'CDP = {np.round(CDP,6)}')
    
    ppr = np.real(pp)
    plt.plot(ppr, 1-Cp,label='1-Cp')
    plt.legend()
    plt.show()
