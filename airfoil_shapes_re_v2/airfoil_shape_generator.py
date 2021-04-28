#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 13:55:39 2020

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt

def joukowsky(IM,t11,t2):
    t1 = t11 * 10.0
    t2 = t2
    T = (t1 + t2) / 100.0    
    kam1 = kam1 * 10    
    kam2 = kam2    
    Kam = (Kam1 + Kam2) / 100.0
    
    bb = 1 / 4.0
    FF = T / (3*3**0.5/4.0)
    GG = 2.0 * Kam    
    fj = bb * FF    
    g = bb * GG
    
    for i in range(IM):
        TETA[i] = 2 * np.pi - (2 * np.pi / (IM - 1)) * (i - 1)
        xx[i] = 2 * bb * np.cos(TETA[i]) + 0.5        
        yy[i] = 2 * fj * (1 - np.cos(TETA[i])) * np.sin(TETA[i]) + 2 * g * (np.sin(TETA[i]))**2

    xx[0] = 1.0
    xx[IM-1] = 1.0
    yy[0] = 0.0
    yy[IM-1] = 0.0
    
#%%    
def naca4(IM,mm,pp,t11,t2):
#    IM: total number
#    NACA ABCD
#    A:mm
#    B:pp
#    C:t11
#    D:t2
    
    NP = int((IM - 1)/2 + 1)
    
    dTeta = 2 * np.pi / (IM - 1)
    
    m = mm/100
    p = pp/10
    t1 = t11*10
    t2 = t2
    T = (t1 + t2)/100
    
    X = np.zeros(NP)
    yc = np.zeros(NP)
    dyc = np.zeros(NP)
    yt = np.zeros(NP)
    vq = np.zeros(NP)
    xu = np.zeros(NP)
    xl = np.zeros(NP)
    yu = np.zeros(NP)
    yl = np.zeros(NP)
    xx = np.zeros(2*NP-1)
    yy = np.zeros(2*NP-1)
    
    for i in range(NP):
        X[i]  = (1.0+np.cos(np.pi - dTeta*i))/2.0
        
        if X[i] < p:
            yc[i] = m / p**2 * (2 * p * X[i] - X[i]**2)
            dyc[i] = m / p**2 * (2 * p - 2 * X[i])
        elif p <= X[i] and X[i] <= 1.0:
            yc[i] = m / (1 - p)**2 * (1 - 2 * p + 2 * p * X[i] - X[i]**2)
            dyc[i] = m / (1 - p)**2 * (2 * p - 2 * X[i])
        
        yt[i] = T / 0.2 * (0.2969 * X[i]**0.5 - 0.126 * X[i] - 0.3516 * X[i]**2 + 0.2843 * X[i]**3 - 0.1015 * X[i]**4)
    

        vq[i] = np.arctan(dyc[i])
        
        xu[i] = X[i] - yt[i] * np.sin(vq[i])
        xl[i] = X[i] + yt[i] * np.sin(vq[i])
        
        yu[i] = yc[i] + yt[i] * np.cos(vq[i])    
        yl[i] = yc[i] - yt[i] * np.cos(vq[i])
    
    xu[0] = 0
    xu[NP-1] = 1
    xl[0] = 0
    xl[NP-1] = 1
#    yu[0] = 0
#    yu[NP-1] = 0
#    yl[0] = 0
#    yl[NP-1] = 0
    
    xx = np.hstack((np.flip(xl), xu[1:]))
    yy = np.hstack((np.flip(yl), yu[1:]))
    
#    xue, xle = np.zeros((NP+2)), np.zeros((NP+2))
#    yue, yle = np.zeros((NP+2)), np.zeros((NP+2))
#    
#    xue[1:-1] = xu
#    xle[1:-1] = xl
#    yue[1:-1] = yu
#    yle[1:-1] = yl
#
#    
#    for i in range(1,NP+1):
#        print(i)
#        xx[i-1] = xle[NP + 1 - i]
#        xx[NP-1+i-1] = xue[i]
#        yy[i-1] = yle[NP + 1 - i]
#        yy[NP-1+i-1] = yue[i]
    
    return xx, yy

xx, yy  = naca4(201,2,0,1,2)
plt.plot(xx,yy)
plt.show()

#%%
def naca210(IM,t11,t2):
#    IM: total number
#    NACA 210CD
#    C:t11
#    D:t2

    m = 0.058
    p = 0.05
    k1 = 361.4
    NP = int((IM - 1)/2 + 1)
    t1 = t11 * 10
    t2 = t2
    T = (t1 + t2) / 100
    
    dTeta = 2 * np.pi / (IM - 1)
    
    X = np.zeros(NP)
    yc = np.zeros(NP)
    dyc = np.zeros(NP)
    yt = np.zeros(NP)
    vq = np.zeros(NP)
    xu = np.zeros(NP)
    xl = np.zeros(NP)
    yu = np.zeros(NP)
    yl = np.zeros(NP)
    xx = np.zeros(2*NP-1)
    yy = np.zeros(2*NP-1)
    
    for i in range(NP):
        X[i]  = (1.0+np.cos(np.pi - dTeta*i))/2.0
        
        if X[i] < p:
            yc[i] = k1 / 6 * (X[i]**3 - 3 * m * X[i]**2 + m**2 * (3 - m) * X[i])
            dyc[i] = k1 / 6 * (3 * X[i]**2 - 6 * m * X[i] + m**2 * (3 - m))
        elif p <= X[i] and X[i] <= 1.0:
            yc[i] = k1 / 6 * m**3 * (1 - X[i])
            dyc[i] = -k1 / 6 * m**3
        
        yt[i] = T / 0.2 * (0.2969 * X[i]**0.5 - 0.126 * X[i] - 0.3516 * X[i]**2 + 0.2843 * X[i]**3 - 0.1015 * X[i]**4)
    
        vq[i] = np.arctan(dyc[i])
        
        xu[i] = X[i] - yt[i] * np.sin(vq[i])
        xl[i] = X[i] + yt[i] * np.sin(vq[i])
        
        yu[i] = yc[i] + yt[i] * np.cos(vq[i])    
        yl[i] = yc[i] - yt[i] * np.cos(vq[i])
    
    xu[0] = 0
    xu[NP-1] = 1
    xl[0] = 0
    xl[NP-1] = 1
#    yu[0] = 0
#    yu[NP-1] = 0
#    yl[0] = 0
#    yl[NP-1] = 0
    
    xx = np.hstack((np.flip(xl), xu[1:]))
    yy = np.hstack((np.flip(yl), yu[1:]))
       
    return xx, yy

xx, yy  = naca210(201,1,2)
plt.plot(xx,yy)
plt.show()

#%%
def naca220(IM,t11,t2):
    m = 0.126
    p = 0.1
    k1 = 51.64
    NP = int((IM - 1)/2 + 1)
    t1 = t11 * 10
    t2 = t2    
    T = (t1 + t2) / 100
    
    dTeta = 2 * np.pi / (IM - 1)
    
    X = np.zeros(NP)
    yc = np.zeros(NP)
    dyc = np.zeros(NP)
    yt = np.zeros(NP)
    vq = np.zeros(NP)
    xu = np.zeros(NP)
    xl = np.zeros(NP)
    yu = np.zeros(NP)
    yl = np.zeros(NP)
    xx = np.zeros(2*NP-1)
    yy = np.zeros(2*NP-1)
    
    for i in range(NP):
        X[i]  = (1.0+np.cos(np.pi - dTeta*i))/2.0
        
        if X[i] < p:
            yc[i] = k1 / 6 * (X[i]**3 - 3 * m * X[i]**2 + m**2 * (3 - m) * X[i])
            dyc[i] = k1 / 6 * (3 * X[i]**2 - 6 * m * X[i] + m**2 * (3 - m))
        elif p <= X[i] and X[i] <= 1.0:
            yc[i] = k1 / 6 * m**3 * (1 - X[i])
            dyc[i] = -k1 / 6 * m**3
        
        yt[i] = T / 0.2 * (0.2969 * X[i]**0.5 - 0.126 * X[i] - 0.3516 * X[i]**2 + 0.2843 * X[i]**3 - 0.1015 * X[i]**4)
    
        vq[i] = np.arctan(dyc[i])
        
        xu[i] = X[i] - yt[i] * np.sin(vq[i])
        xl[i] = X[i] + yt[i] * np.sin(vq[i])
        
        yu[i] = yc[i] + yt[i] * np.cos(vq[i])    
        yl[i] = yc[i] - yt[i] * np.cos(vq[i])
    
    xu[0] = 0
    xu[NP-1] = 1
    xl[0] = 0
    xl[NP-1] = 1
#    yu[0] = 0
#    yu[NP-1] = 0
#    yl[0] = 0
#    yl[NP-1] = 0
    
    xx = np.hstack((np.flip(xl), xu[1:]))
    yy = np.hstack((np.flip(yl), yu[1:]))
        
    return xx, yy

def naca230(IM,t11,t2):
    m = 0.2025
    p = 0.15
    k1 = 15.957
    NP = int((IM - 1)/2 + 1)   
    t1 = t11 * 10
    t2 = t2    
    T = (t1 + t2) / 100
    
    dTeta = 2 * np.pi / (IM - 1)
    
    X = np.zeros(NP)
    yc = np.zeros(NP)
    dyc = np.zeros(NP)
    yt = np.zeros(NP)
    vq = np.zeros(NP)
    xu = np.zeros(NP)
    xl = np.zeros(NP)
    yu = np.zeros(NP)
    yl = np.zeros(NP)
    xx = np.zeros(2*NP-1)
    yy = np.zeros(2*NP-1)
    
    for i in range(NP):
        X[i]  = (1.0+np.cos(np.pi - dTeta*i))/2.0
        
        if X[i] < p:
            yc[i] = k1 / 6 * (X[i]**3 - 3 * m * X[i]**2 + m**2 * (3 - m) * X[i])
            dyc[i] = k1 / 6 * (3 * X[i]**2 - 6 * m * X[i] + m**2 * (3 - m))
        elif p <= X[i] and X[i] <= 1.0:
            yc[i] = k1 / 6 * m**3 * (1 - X[i])
            dyc[i] = -k1 / 6 * m**3
            
        yt[i] = T / 0.2 * (0.2969 * X[i]**0.5 - 0.126 * X[i] - 0.3516 * X[i]**2 + 0.2843 * X[i]**3 - 0.1015 * X[i]**4)
        
        vq[i] = np.arctan(dyc[i])
        
        xu[i] = X[i] - yt[i] * np.sin(vq[i])
        xl[i] = X[i] + yt[i] * np.sin(vq[i])
        
        yu[i] = yc[i] + yt[i] * np.cos(vq[i])    
        yl[i] = yc[i] - yt[i] * np.cos(vq[i])
    
    xu[0] = 0
    xu[NP-1] = 1
    xl[0] = 0
    xl[NP-1] = 1
#    yu[0] = 0
#    yu[NP-1] = 0
#    yl[0] = 0
#    yl[NP-1] = 0
    
    xx = np.hstack((np.flip(xl), xu[1:]))
    yy = np.hstack((np.flip(yl), yu[1:]))
       
    return xx, yy

def naca240(IM,t11,t2):
    m = 0.29
    p = 0.2
    k1 = 6.643
    NP = int((IM - 1)/2 + 1) 
    t1 = t11 * 10
    t2 = t2    
    T = (t1 + t2) / 100
    
    dTeta = 2 * np.pi / (IM - 1)
    
    X = np.zeros(NP)
    yc = np.zeros(NP)
    dyc = np.zeros(NP)
    yt = np.zeros(NP)
    vq = np.zeros(NP)
    xu = np.zeros(NP)
    xl = np.zeros(NP)
    yu = np.zeros(NP)
    yl = np.zeros(NP)
    xx = np.zeros(2*NP-1)
    yy = np.zeros(2*NP-1)
    
    for i in range(NP):
        X[i]  = (1.0+np.cos(np.pi - dTeta*i))/2.0
        
        if X[i] < p:
            yc[i] = k1 / 6 * (X[i]**3 - 3 * m * X[i]**2 + m**2 * (3 - m) * X[i])
            dyc[i] = k1 / 6 * (3 * X[i]**2 - 6 * m * X[i] + m**2 * (3 - m))
        elif p <= X[i] and X[i] <= 1.0:
            yc[i] = k1 / 6 * m**3 * (1 - X[i])
            dyc[i] = -k1 / 6 * m**3
        
        yt[i] = T / 0.2 * (0.2969 * X[i]**0.5 - 0.126 * X[i] - 0.3516 * X[i]**2 + 0.2843 * X[i]**3 - 0.1015 * X[i]**4)
    
        vq[i] = np.arctan(dyc[i])
        
        xu[i] = X[i] - yt[i] * np.sin(vq[i])
        xl[i] = X[i] + yt[i] * np.sin(vq[i])
        
        yu[i] = yc[i] + yt[i] * np.cos(vq[i])    
        yl[i] = yc[i] - yt[i] * np.cos(vq[i])
    
    xu[0] = 0
    xu[NP-1] = 1
    xl[0] = 0
    xl[NP-1] = 1
#    yu[0] = 0
#    yu[NP-1] = 0
#    yl[0] = 0
#    yl[NP-1] = 0
    
    xx = np.hstack((np.flip(xl), xu[1:]))
    yy = np.hstack((np.flip(yl), yu[1:]))
    
    return xx, yy

def naca250(IM,t11,t2):
    m = 0.391
    p = 0.25
    k1 = 3.23
    NP = int((IM - 1)/2 + 1)   
    t1 = t11 * 10
    t2 = t2    
    T = (t1 + t2) / 100
    
    dTeta = 2 * np.pi / (IM - 1)
    
    X = np.zeros(NP)
    yc = np.zeros(NP)
    dyc = np.zeros(NP)
    yt = np.zeros(NP)
    vq = np.zeros(NP)
    xu = np.zeros(NP)
    xl = np.zeros(NP)
    yu = np.zeros(NP)
    yl = np.zeros(NP)
    xx = np.zeros(2*NP-1)
    yy = np.zeros(2*NP-1)
    
    for i in range(NP):
        X[i]  = (1.0+np.cos(np.pi - dTeta*i))/2.0
        
        if X[i] < p:
            yc[i] = k1 / 6 * (X[i]**3 - 3 * m * X[i]**2 + m**2 * (3 - m) * X[i])
            dyc[i] = k1 / 6 * (3 * X[i]**2 - 6 * m * X[i] + m**2 * (3 - m))
        elif p <= X[i] and X[i] <= 1.0:
            yc[i] = k1 / 6 * m**3 * (1 - X[i])
            dyc[i] = -k1 / 6 * m**3
        
        yt[i] = T / 0.2 * (0.2969 * X[i]**0.5 - 0.126 * X[i] - 0.3516 * X[i]**2 + 0.2843 * X[i]**3 - 0.1015 * X[i]**4)
    
        vq[i] = np.arctan(dyc[i])
        
        xu[i] = X[i] - yt[i] * np.sin(vq[i])
        xl[i] = X[i] + yt[i] * np.sin(vq[i])
        
        yu[i] = yc[i] + yt[i] * np.cos(vq[i])    
        yl[i] = yc[i] - yt[i] * np.cos(vq[i])
    
    xu[0] = 0
    xu[NP-1] = 1
    xl[0] = 0
    xl[NP-1] = 1
#    yu[0] = 0
#    yu[NP-1] = 0
#    yl[0] = 0
#    yl[NP-1] = 0
    
    xx = np.hstack((np.flip(xl), xu[1:]))
    yy = np.hstack((np.flip(yl), yu[1:]))
        
    return xx, yy


#%%
N = 201
    
C_list = [0]
D_list = [6,8]    

naca4_generate = False
if naca4_generate: 
    for C in C_list:
        for D in D_list:
            A = 0
            B = 0
            xx, yy  = naca4(N,A,B,C,D)                
            airfoil = np.zeros((N,2))
            airfoil[:,0], airfoil[:,1] = xx, yy
            np.savetxt(f'naca{A}{B}{C}{D}.txt', airfoil)

C_list = [1]
D_list = [0,2,4,6]    

naca4_generate = False
if naca4_generate: 
    for C in C_list:
        for D in D_list:
            A = 0
            B = 0
            xx, yy  = naca4(N,A,B,C,D)                
            airfoil = np.zeros((N,2))
            airfoil[:,0], airfoil[:,1] = xx, yy
            np.savetxt(f'naca{A}{B}{C}{D}.txt', airfoil)
        
#%%
A_list = [2,3,4,5,6]
B_list = [2,3,4,5,6]
C_list = [0]
D_list = [6,8]
    
naca4_generate = False
if naca4_generate:    
    for A in A_list:
        for B in B_list:
            for C in C_list:
                for D in D_list:
                    xx, yy  = naca4(N,A,B,C,D)                
                    airfoil = np.zeros((N,2))
                    airfoil[:,0], airfoil[:,1] = xx, yy
                    np.savetxt(f'naca{A}{B}{C}{D}.txt', airfoil)

#%%                    
C_list = [1]
D_list = [0,2,4,6]
    
naca4_generate = False
if naca4_generate:    
    for A in A_list:
        for B in B_list:
            for C in C_list:
                for D in D_list:
                    xx, yy  = naca4(N,A,B,C,D)                
                    airfoil = np.zeros((N,2))
                    airfoil[:,0], airfoil[:,1] = xx, yy
                    np.savetxt(f'naca{A}{B}{C}{D}.txt', airfoil)
            
#%%
naca210_generate = False    
for C in C_list:
    for D in D_list:    
        xx, yy  = naca210(N,C,D)
        airfoil = np.zeros((N,2))
        airfoil[:,0], airfoil[:,1] = xx, yy
        np.savetxt(f'naca210{C}{D}.txt', airfoil)

#%%
naca220_generate = False    
for C in C_list:
    for D in D_list:    
        xx, yy  = naca220(N,C,D)
        airfoil = np.zeros((N,2))
        airfoil[:,0], airfoil[:,1] = xx, yy
        np.savetxt(f'naca220{C}{D}.txt', airfoil)

#%%
C_list = [1]
D_list = [0,2,4,6,8]
        
naca230_generate = True    
for C in C_list:
    for D in D_list:    
        xx, yy  = naca230(N,C,D)
        airfoil = np.zeros((N,2))
        airfoil[:,0], airfoil[:,1] = xx, yy
        np.savetxt(f'naca230{C}{D}.txt', airfoil)

C_list = [2]
D_list = [0,2,4]
        
naca230_generate = True    
for C in C_list:
    for D in D_list:    
        xx, yy  = naca230(N,C,D)
        airfoil = np.zeros((N,2))
        airfoil[:,0], airfoil[:,1] = xx, yy
        np.savetxt(f'naca230{C}{D}.txt', airfoil)

#%%
C_list = [1]
D_list = [0,2,4,6,8]

naca240_generate = False    
for C in C_list:
    for D in D_list:    
        xx, yy  = naca240(N,C,D)
        airfoil = np.zeros((N,2))
        airfoil[:,0], airfoil[:,1] = xx, yy
        np.savetxt(f'naca240{C}{D}.txt', airfoil)        

#%%
naca250_generate = False    
for C in C_list:
    for D in D_list:    
        xx, yy  = naca250(N,C,D)
        airfoil = np.zeros((N,2))
        airfoil[:,0], airfoil[:,1] = xx, yy
        np.savetxt(f'naca250{C}{D}.txt', airfoil)                