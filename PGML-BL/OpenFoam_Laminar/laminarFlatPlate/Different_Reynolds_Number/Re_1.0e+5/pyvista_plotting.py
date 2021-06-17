#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 13:45:40 2021

@author: suraj
"""

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from pyvista import examples

filename = f'../../../../Similarity_Solution/su2.csv'
data_blasisus = np.genfromtxt(filename, delimiter=',')

## grid is the central object in VTK where every field is added on to grid
grid = pv.UnstructuredGrid('./VTK/Re_1.0e+5_10000.vtk')

grid_x = pv.UnstructuredGrid('./VTK/Re_1.0e+5_10000.vtk')

#%%
centers = grid.cell_centers()

cell_centes = np.array(centers.points)

U_cell = np.array(grid.cell_arrays['U'])
P_cell = np.array(grid.cell_arrays['p'])

#%%
pp = grid.points
pp_slice = grid.points[:,0] >= 0.0
grid_clipped_points = pp[pp_slice]
pp_slice = grid_clipped_points[:,0] <= 0.05
grid_clipped_points = grid_clipped_points[pp_slice]

#%%
dataset = examples.download_office()
mesh = examples.load_airplane()

bounds = [-0.025,0.0, 0,0.065, 0,0.002]
clipped = grid.clip_box(bounds)

bounds = [0.1,0.15, 0,0.065, 0,0.002]
clipped = clipped.clip_box(bounds)
#
bounds = [0.0,0.15, 0.01,0.06, 0,0.002]
clipped = clipped.clip_box(bounds)

centers_clipped = np.array(clipped.cell_centers().points)

#%%
p = pv.Plotter()
#p.add_mesh(grid, show_edges=True, color='white', label='Input')
p.add_mesh(clipped,  show_edges=True, color='red', label='Input')
p.add_mesh(clipped, scalars='U', opacity=1.0)
p.show(cpos='xy')

#%%
lx = 0.1
nx = 200
ly = 0.02
ny = 400
x = np.linspace(0,lx,nx+1)
y = np.linspace(0,ly,ny+1)

u_interpolated = np.zeros((ny+1,nx+1))
v_interpolated = np.zeros((ny+1,nx+1))
p_interpolated = np.zeros((ny+1,nx+1))

for i in range(nx+1):
    a = [x[i],0.0,0.0005]
    b = [x[i],ly,0.0005]
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
ax.contour(X,Y, u_interpolated, colors='k')

    
ax.set_ylim([0,0.005])
#ax.set_aspect(1)
plt.show()

#%%
fig, ax = plt.subplots(1,2,figsize=(10,5))

for x in [0.005,0.015,0.025,0.03]:
    a = [x,0.0,0.0005]
    b = [x,0.005,0.0005]
    line = pv.Line(a, b, resolution=200)
    
    result = grid.probe(line)
    result_c = clipped.probe(line)
   
    y = np.array(line.points)[:,1]    
    ux = result['U'][:,0]
    uy = result['U'][:,1]
    
    U0 = 1.0
    nu = 1.0e-06
    
    y = y*np.sqrt(U0/(nu*x))
    ux = ux/U0
        
    ax[0].plot(ux, y, '-', lw=2, label=f'x={x}')
    ax[1].plot(uy, y, 'o-', lw=2, label=f'x={x}')
    
    y_c = np.array(line.points)[:,1]    
    ux_c = result_c['U'][:,0]
    uy_c = result_c['U'][:,1]
    
    U0 = 1.0
    nu = 1.0e-06
    
    y_c = y_c*np.sqrt(U0/(nu*x))
    ux_c = ux_c/U0
        
    ax[0].plot(ux_c, y_c, '--', lw=2, label=f'x={x}')
    ax[1].plot(uy_c, y_c, 's--', lw=2, label=f'x={x}')

ax[0].plot(data_blasisus[:,0], data_blasisus[:,1], ls='none', 
           marker='o', color='k', ms=8, fillstyle='none', label='Blasius')
    
ax[0].legend()
ax[0].grid()
ax[0].set_xlim([0,1.1])
ax[0].set_ylim([0,6])
ax[0].set_xlabel('$u/u_\infty$')

ax[1].legend()
ax[1].grid()
ax[1].set_ylim([0,5])
ax[1].set_xlabel('$v/u_\infty$')
plt.show()
fig.savefig('trial.png', dpi=300)

#%%
U = grid['U']
contours = grid.contour(360, scalars='U')

p = pv.Plotter(window_size=[600,400])
p.set_background(color='white')
p.add_mesh(grid, scalars='U', opacity=1.0)
p.add_mesh(contours, color='k')
#p.add_mesh(contours, color="white", line_width=5)
p.show(cpos='xy',)
#p.save_graphic('u1.png')
p.screenshot('u.png')


#%%

p = pv.Plotter(window_size=[1024,512])
p.set_background(color='white')
p.add_mesh(grid, show_edges=False, opacity=0.5, line_width=0.5, color='white')
p.add_mesh(line, color="b")
#p.add_mesh(centers, color="r", point_size=4.0, render_points_as_spheres=True)
p.show(cpos='xy', screenshot="mesh.png")
p.screenshot('mesh.png')

#%%
p = pv.Plotter()
p.set_background(color='white')
p.add_mesh(grid, color='white', show_edges=True,)
p.show(cpos='xy', screenshot="mesh.png")
p.screenshot('mesh.png')


grid.plot(cpos='xy', screenshot="mesh.png", color='white', show_edges=True,)

#%%
grid.plot_over_line(a, b, resolution=100)