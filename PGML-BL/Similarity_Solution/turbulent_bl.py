"""
Created on Sun Jun 20 15:29:55 2021

@author: suraj
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint

Lx = 2.0
nx = 500
ny = 500
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


        
#%%
X, Y = np.meshgrid(x,y)
slice_ = np.where(Y<=delta, Y/delta, 1)

#%%
u_power = uinf*(slice_)**(1/7)

#%%
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

# Shape factor
H = delta_s/delta_s

rho = 1.0

# Shear Stress
tw 	= u_s**2*rho

# Cf
cf3 = tw/ (0.5*rho*ue**2)
        
#%%

cf_weighardt = 0.288*(np.log10(Rex))**(-2.45)
plt.plot(Rex[1:], cf1.flatten()[1:],label='1/5')
plt.plot(Rex[1:], cf2.flatten()[1:],label='1/4')
plt.plot(Rex[1:], cf3.flatten()[1:],label='log')
plt.plot(Rex[1:], cf_weighardt.flatten()[1:], 'k--', label='Weighardt')
plt.ylim([0,0.008])
plt.legend()
plt.show()

#%%
slice_ = np.where(Y<=delta_log, Y/delta_log, 1)
u_log = uinf*(slice_)**(1/7)
        
#%%
plt.contourf(X,Y,u_power,40,cmap='viridis')
plt.plot(x,delta.flatten(),'k')
plt.plot(x,delta_log.flatten(),'k--')
plt.ylim([0,3.0*delta[0,-1]])
plt.colorbar()
plt.show()

#%%
plt.plot(u_power[:,-1],Y[:,-1],label='1/7 Power law')
plt.plot(u_log[:,-1],Y[:,-1], label='Log wake')
plt.ylim([0,3.0*delta[0,-1]])
plt.legend()
plt.grid()
plt.show()
