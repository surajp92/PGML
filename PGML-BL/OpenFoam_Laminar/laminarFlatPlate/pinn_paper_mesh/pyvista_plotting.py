#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 13:45:40 2021

@author: suraj
"""

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

filename = f'../../../Similarity_Solution/su2.csv'
data_blasisus = np.genfromtxt(filename, delimiter=',')

## grid is the central object in VTK where every field is added on to grid
grid = pv.UnstructuredGrid('./VTK/pinn_paper_mesh_10000.vtk')

centers = grid.cell_centers()

cell_centes = np.array(centers.points)

U_cell = np.array(grid.cell_arrays['U'])
P_cell = np.array(grid.cell_arrays['p'])

fig, ax = plt.subplots(1,2,figsize=(10,5))

for x in [0.005,0.015,0.025,0.03]:
    a = [x,0.0,0.0005]
    b = [x,0.005,0.0005]
    line = pv.Line(a, b, resolution=200)
    
    result = grid.probe(line)
   
    y = np.array(line.points)[:,1]
    ux = result['U'][:,0]
    uy = result['U'][:,1]
    
    U0 = 1.0
    nu = 1.0e-06
    
    y = y*np.sqrt(U0/(nu*x))
    ux = ux/U0
        
    ax[0].plot(ux, y, '-', lw=2, label=f'x={x}')
    ax[1].plot(uy, y, 'o-', lw=2, label=f'x={x}')

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

#%%

p = pv.Plotter(window_size=[1024,512])
p.add_mesh(grid, show_edges=False, opacity=0.5, line_width=0.5, color='white')
p.add_mesh(line, color="b")
#p.add_mesh(centers, color="r", point_size=4.0, render_points_as_spheres=True)
p.show(cpos='xy', screenshot="mesh.png")
#p.save('mesh.png')

#%%
grid.plot(cpos='xy', screenshot="mesh", color='white', show_edges=True,)

grid.plot_over_line(a, b, resolution=100)