#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:28:41 2020

@author: suraj
"""
import numpy as np
import pandas as pd
import csv

num_xy = 201
num_cp = num_xy - 1
  
db = 'train_data_naca230_v1.csv'
df = pd.read_csv(db, encoding='utf-8')

col_name = []
col_name.append('Airfoil')
for i in range(num_xy):
    col_name.append(f'x{i}')
for i in range(num_xy):
    col_name.append(f'y{i}')
col_name.append('AOA1')    
col_name.append('AOA2')
col_name.append('RE')
col_name.append('CL')
col_name.append('CD')
col_name.append('CM')
col_name.append('CPmin')

df.columns = col_name

db = 'train_data_naca240.csv'
df2 = pd.read_csv(db, encoding='utf-8')
df2.columns = col_name
df = df.append(df2, ignore_index=True)

df.to_csv(r'test_data_v1.csv', index = False, na_rep='nan')
