#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 12:42:32 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, quad

font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

#%%
def rk4(y0,t0,dt):
    """Execute a Runge-Kutta step
	Keyword arguments:
	g -- integrand function
	y0 -- (list) initial data
	t0 -- (float) initial time
	dt -- (float) time step
	Return :
	y1 -- list of components of the function at time t1
	"""
    c1 = rhs(y0        , t0        )*dt
    c2 = rhs(y0 + c1/2., t0 + dt/2.)*dt
    c3 = rhs(y0 + c2/2., t0 + dt/2.)*dt
    c4 = rhs(y0 + c3   , t0 + dt   )*dt
    y1 = y0 + 1./6.*( c1 + 2.*c2 + 2.*c3 + c4 )
    return y1
    
def rhs(y,t):
    r = np.zeros_like(y)
    r[0] = -0.5*y[0]*y[2]
    r[1] = y[0]
    r[2] = y[1]
    
    return r

#%%
ne = 3
y = np.zeros(ne)

n = 30
a = 0.0
b = 10.0
h = (b-a)/n

# boundary conditions
y2a = 0.0
y3a = 0.0
y2b = 1.0

# initial two guesses for y(1) at x=0
A1 = 1.0
A2 = 0.5

# first guess: initial condition
y[0] = A1 # guesss
y[1] = y2a # BC
y[2] = y3a # BC

t = np.linspace(a,b,n+1)
X = odeint(rhs, y, t)

plt.plot(X[:,0])
plt.plot(X[:,1])
plt.plot(X[:,2])
plt.show()

#%%
for k in range(1,n+1):
    x = (k-1)*h
    y = rk4(y, x, h)

# assign estimated value for y(2) at x=10
B1 = y[1]        

#%%
# second guess: initial condition
y[0] = A2 # guesss
y[1] = y2a # BC
y[2] = y3a # BC

for k in range(1,n+1):
    x = (k-1)*h
    y = rk4(y, x, h)

# assign estimated value for y(2) at x=10
B2 = y[1]

#%%
for i in range(50):
    guess = A2 + (y2b - B2)/((B2-B1)/(A2-A1)) # secant method
    
    y[0] = guess # guesss
    y[1] = y2a # BC
    y[2] = y3a # BC

    for k in range(1,n+1):
        x = (k-1)*h
        y = rk4(y, x, h)

    B1 = B2
    B2 = y[1]
    A1 = A2
    A2 = guess
    
    print('%1d %1.4f' % (i, B2))
    
    if abs(B2 - y2b) <= 1e-10:
        break
    
#%%
uu = np.zeros(n+1)
eta = np.zeros(n+1)
yy = np.zeros((n+1,ne))

y[0] = guess # guesss
y[1] = y2a # BC
y[2] = y3a # BC

yy[0,:] = y

t = np.linspace(a,b,n+1)
X = odeint(rhs, y, t)

for k in range(1,n+1):
    x = (k-1)*h
    y = rk4(y, x, h)
    uu[k] = y[1]
    eta[k] = k*h
    yy[k,:] = y

#%%    
fig, ax = plt.subplots(1,1,figsize=(6,5))

ax.plot(eta, yy[:,0], lw=3, label='$f_1$')
ax.plot(eta, yy[:,1], lw=3, label='$f_2$')
ax.plot(eta, yy[:,2], lw=3, label='$f_3$')

# ax.plot(eta, X[:,0], '--')
# ax.plot(eta, X[:,1], '--')
# ax.plot(eta, X[:,2], '--')

ax.legend()

ax.set_xlim([0,6])
ax.set_ylim([-0.5,2])
ax.grid()
plt.show()
fig.tight_layout()
fig.savefig('variables.png', dpi=300)


#%%
nu = 0.0000171 # kinematic viscosity
uinf = 1.0
length = 1.
nx = 100

xx = np.zeros(nx+1)
Rex= np.zeros(nx+1)
yy = np.zeros((nx+1, n+1))

xx = np.linspace(0, length, nx+1)
Rex = xx*uinf/nu

yy = eta.reshape(1,-1)*np.sqrt(nu*xx.reshape(-1,1)/uinf)

xxm = np.zeros((nx+1, n+1))
yym = np.zeros((nx+1, n+1))
uum = np.zeros((nx+1, n+1))
vvm = np.zeros((nx+1, n+1))

for j in range(n+1):
    for i in range(nx+1):
        xxm[i,j] = xx[i]
        yym[i,j] = yy[i,j]
        uum[i,j] = uu[j]
        vvm[i,j] = 0.0

#%%
        
#data = np.genfromtxt('blasius-schlichting-dataset.csv', delimiter=',')
data = np.genfromtxt('su2.csv', delimiter=',')
        
fig, ax = plt.subplots(1,1,figsize=(4,5))
ax.plot(uu, eta, 'k-', lw = 2)
ax.plot(data[:,0], data[:,1], ls='None', 
        marker='o', color='r', ms=8, fillstyle='none')
ax.grid()
ax.set_xlim([0,1.1])
ax.set_ylim([0,10])
plt.show()

#%%        
fig, ax = plt.subplots(1,1,figsize=(8,5))
cs = ax.contourf(xxm,yym,uum,120,cmap='jet',vmin=0,vmax=1)
cbar = fig.colorbar(cs, ax=ax)
cbar.set_label('$u/u_\infty$', rotation=90, labelpad=15)

plt.show()
fig.tight_layout()
fig.savefig('contour.png', dpi=300)

#%%
from matplotlib.collections import LineCollection

x = xxm
y = yym
#x, y = np.meshgrid(np.linspace(0,1, 11), np.linspace(0, 0.6, 7))

# plt.scatter(x, y)

segs1 = np.stack((x,y), axis=2)
segs2 = segs1.transpose(1,0,2)
plt.gca().add_collection(LineCollection(segs1))
plt.gca().add_collection(LineCollection(segs2))

plt.xlim([0.0, 0.3])
plt.ylim([0.0, 0.02])

plt.show()
