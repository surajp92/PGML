#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 14:22:27 2021

@author: suraj
"""

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from pyvista import examples
from scipy.integrate import odeint

filename = f'../../../Similarity_Solution/su2.csv'
data_blasisus = np.genfromtxt(filename, delimiter=',')

## grid is the central object in VTK where every field is added on to grid
grid = pv.UnstructuredGrid('./VTK/yplus_+5_5000.vtk')

#%%
centers = grid.cell_centers()
cell_centes = np.array(centers.points)

U_cell = np.array(grid.cell_arrays['U'])
P_cell = np.array(grid.cell_arrays['p'])

#%%
p = pv.Plotter()
#p.add_mesh(grid, show_edges=True, color='white', label='Input')
p.add_mesh(grid,  show_edges=True, color='white', label='Input')
p.add_mesh(grid, scalars='U', opacity=1.0)
p.show(cpos='xy')

#%%
lx = 2.0
nx = 200
ly = 1.0
ny = 400
x = np.linspace(0,lx,nx+1)
y = np.linspace(0,ly,ny+1)

u_interpolated = np.zeros((ny+1,nx+1))
v_interpolated = np.zeros((ny+1,nx+1))
p_interpolated = np.zeros((ny+1,nx+1))

for i in range(nx+1):
    a = [x[i],0.0,0.05]
    b = [x[i],ly,0.05]
    line = pv.Line(a, b, resolution=ny)
    
    result = grid.probe(line)
    
    y = np.array(line.points)[:,1]    
    ux = result['U'][:,0]
    uy = result['U'][:,1]
    p = result['p']
    
    u_interpolated[:,i] = ux
    v_interpolated[:,i] = uy
    p_interpolated[:,i] = p
    
X,Y = np.meshgrid(x,y)

np.savez('data.npz', u = u_interpolated,
         v = v_interpolated, 
         p = p_interpolated,
         x=x, y=y,
         X=X, Y=Y         
         )

#%%
fig, ax = plt.subplots(1,1,figsize=(10,5))
cs = ax.contourf(X,Y, u_interpolated, 120, cmap='jet')
fig.colorbar(cs, ax = ax)
#ax.contour(X,Y, u_interpolated, 10, colors='k')
    
ax.set_ylim([0,0.1])
#ax.set_aspect(1)
plt.show()

#%%
Lx = 2.0
nx = 500
ny = 1000
x = np.linspace(0,Lx,nx+1)
x[0] = 1.0e-5
uinf = 69.4
nu = 1.388e-5

Rex = uinf*x/nu

delta = 0.38*x/(Rex**(1/5))

Ly = 1.0 #delta[-1]*10
y = np.linspace(0,Ly,ny+1)

delta = delta.reshape([1,-1])

# Boundary layer displacement thichkness with x
# eta- f when eta = inf
delta_s = 0.048  *x/(Rex**(1/5))

# Boundary layer momentum thichkness with x
delta_ss = 0.037  *x/(Rex**(1/5))

# Shape factor
H = delta_s/delta_ss

# Friction coefficient with x
cf1 = 0.059 *(nu/(uinf*delta))**(1/5)
cf2 = 0.0456 *(nu/(uinf*delta))**(1/4)

X, Y = np.meshgrid(x,y)
slice_ = np.where(Y<=delta, Y/delta, 1)

# seventh power
u_power = uinf*(slice_)**(1/7)

# log wake 
B = 5.3
A = 0.62
k = 0.41

def fturbulent (Red,Rex,k,B,A):
    return  1/((1/k)*np.log(Red)+B-A)**2


def fu_s (Red,Rex,k,B,A):
    return  1/((1/k)*np.log(Red)+B-A)

Re_x = np.copy(Rex)
ue = uinf

## Initial condition approximation
Re_delta0 = (x[0]*0.048/Re_x[0]**(1/5))*ue/nu

## Differential equation for turbulent boundary layers
Re_delta = odeint(fturbulent,Re_delta0,Re_x,args=(k,B,A,),tcrit=[x[0]])
u_s   = ue * fu_s (Re_delta.flatten(),Re_x,k,B,A)

# Boundary layer displacement thichkness with x
delta_s  = nu*Re_delta.flatten()/ue
delta_ss = delta_s

# Boundary layer thichkness with x
delta_log = ue*delta_s/u_s	

delta_log = delta_log.reshape([1,-1])

slice_log_ = np.where(Y<=delta_log, Y/delta_log, 1)
u_log = uinf*(slice_log_)**(1/7)

# Shape factor
H = delta_s/delta_s

rho = 1.0

# Shear Stress
tw 	= u_s**2*rho

# Cf
cf3 = tw/ (0.5*rho*ue**2)

#%%
fig, ax = plt.subplots(1,1,figsize=(10,5))
cs = ax.contourf(X,Y, u_power, 120, cmap='jet')
fig.colorbar(cs, ax = ax)
#ax.contour(X,Y, u_interpolated, 10, colors='k')
    
ax.set_ylim([0,0.1])
#ax.set_aspect(1)
plt.show()

#%%
fig, ax = plt.subplots(1,2,figsize=(10,5))

   
for xl in [1.0]:
    a = [xl,0.0,0.05]
    b = [xl,0.1,0.05]
    line = pv.Line(a, b, resolution=200)
    
    result = grid.probe(line)
   
    y = np.array(line.points)[:,1]    
    ux = result['U'][:,0]
    uy = result['U'][:,1]
    
#    y = y*np.sqrt(U0/(nu*x))
    ux = ux/uinf
        
    ax[0].plot(ux, y, '-', lw=2, label=f'x={xl}')
    ax[1].plot(uy, y, 'o-', lw=2, label=f'x={xl}')
    
#ax[0].plot(data_blasisus[:,0], data_blasisus[:,1], ls='none', 
#           marker='o', color='k', ms=8, fillstyle='none', label='Blasius')

idx = np.where(x==xl)[0][0]
ax[0].plot(u_power[:,idx]/uinf,Y[:,idx],'--', label='1/7 Power law')
#ax[0].plot(u_log[:,-1]/uinf,Y[:,-1], '--', label='Log wake')
    
ax[0].legend()
ax[0].grid()
ax[0].set_xlim([0,1.1])
ax[0].set_ylim([0,0.04])
ax[0].set_xlabel('$u/u_\infty$')

ax[1].legend()
ax[1].grid()
ax[1].set_ylim([0,0.04])
ax[1].set_xlabel('$v/u_\infty$')
plt.show()
fig.savefig('trial.png', dpi=300)