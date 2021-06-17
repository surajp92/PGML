"""
Created on Wed Jul 29 11:58:00 2020

@author: suraj
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from turbulucid import *

font = {'size'   : 14}    
plt.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = (12, 6)

from os.path import join
import turbulucid
case = Case("./data_slice/data_slice_0_0.vtp")
u = case["U"]

print(case.fields)

#%%
case["magUMean"] = np.linalg.norm(case["U"], axis=1)
case["magUMean"]

h = 0.001
Uref = 1.0
plot_boundaries(case, scaleX=h, scaleY=h)
plot_field(case, "magUMean",scaleX=h, scaleY=h)
# plt.xlim([-0, 10])
# plt.ylim([0, 2])
plt.xlabel(r'$x/h$')
plt.ylabel(r'$y/h$')
plt.tight_layout()
# plt.savefig('contour.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig('contour.png', dpi=300, bbox_inches='tight', pad_inches=0)

#%%
plot_vectors(case, 'U', normalize=True, scaleX=h, scaleY=h,  
             sampleByPlane=True, planeResolution=[10, 10], scale=1)
plt.xlim([-0, 10])
plt.ylim([0, 2])
plt.xlabel(r'$x/h$')
plt.ylabel(r'$y/h$')
plt.tight_layout()
# plt.savefig('vectors.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig('vectors.png', dpi=300, bbox_inches='tight', pad_inches=0)

#%%
h = 0.0127
Uref = 44.2
xh_n4 = np.loadtxt('x_h_-4.csv',delimiter=',')
xh_1 = np.loadtxt('x_h_+1.csv',delimiter=',')
xh_4 = np.loadtxt('x_h_+4.csv',delimiter=',')
xh_6 = np.loadtxt('x_h_+6.csv',delimiter=',')

plot_boundaries(case, scaleX=h, scaleY=h)
plot_field(case, "magUMean",scaleX=h, scaleY=h,) #cmap='YlGn'

plt.xlim([-5, 10])
plt.ylim([0, 4])

hpos = -4*h
point1, point2 = (hpos, 0), (hpos, 1)
points, data = profile_along_line(case, point1, point2)
plt.plot(data['U'][:,0]/Uref + hpos/h, points/h, 'r-',lw=3)
plt.plot(xh_n4[:,2] + hpos/h, xh_n4[:,1], 'ko', ms=8, mew=2,fillstyle='none')

hpos = h
point1, point2 = (hpos, 0), (hpos, 1)
points, data = profile_along_line(case, point1, point2)
plt.plot(data['U'][:,0]/Uref + hpos/h, points/h, 'r-',lw=3)
plt.plot(xh_1[:,2] + hpos/h, xh_1[:,1], 'ko', ms=8, mew=2,fillstyle='none')

hpos = 4*h
point1, point2 = (hpos, 0), (hpos, 1)
points, data = profile_along_line(case, point1, point2)
plt.plot(data['U'][:,0]/Uref + hpos/h, points/h, 'r-',lw=3)
plt.plot(xh_4[:,2] + hpos/h, xh_4[:,1], 'ko', ms=8, mew=2,fillstyle='none')

hpos = 6*h
point1, point2 = (hpos, 0), (hpos, 1)
points, data = profile_along_line(case, point1, point2)
plt.plot(data['U'][:,0]/Uref + hpos/h, points/h, 'r-',lw=3)
plt.plot(xh_6[:,2] + hpos/h, xh_6[:,1], 'ko', ms=8, mew=2,fillstyle='none')

plt.xlabel(r'$x/h$')
plt.ylabel(r'$y/h$')

plt.tight_layout()
plt.savefig('velocity_profile.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
plt.savefig('velocity_profile.png', dpi=300, bbox_inches='tight', pad_inches=0)

#%%

a = xh_1[:,1]
b = xh_1[:,2]

plt.plot(b, a)
plt.show()