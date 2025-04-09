# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 11:06:47 2025

@author: Arjoh
"""

#Library Importation
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose as sd
import matplotlib.pyplot as plt

###--------------Import and Format Data
file_path = r"C:\Users\Arjoh\Downloads\Assignment 7 Data 1.xlsx"
df = pd.read_excel(file_path)
df.dropna(inplace=True)
df.head()

#Plot Data
df.plot(x='t', y='data', style='.')
plt.xlabel("Time (t)")
plt.ylabel("Data")
plt.title("Time Series Data")
plt.show()

###--------------Multiplicative Decomposition
mult_result = sd(df['data'], period = 4, model = 'multiplicative') #, extrapolate_trend= 'freq')
mult_result.plot()
plt.xlabel("t")
plt.title("Multiplicative Model for Data 1")
plt.show()

###--------------Additive Decomposition
add_result = sd(df['data'],period = 4, model = 'additive')
add_result.plot()
plt.xlabel("t")
plt.title("Additive Model for Data 1")
plt.show()

###--------------Prepare Outputs
#Multiplicative
mult_decomposition = pd.DataFrame({
    'Observed Y': mult_result.observed,
    'Ctr Avg_Trend': mult_result.trend,
    'Seasonal': mult_result.seasonal,
    'd(t)': mult_result.trend * mult_result.resid,
    'Residual': mult_result.resid})

print(mult_decomposition.head())
print(mult_decomposition.tail())

#Additive
add_decomposition = pd.DataFrame({
    'Observed Y': add_result.observed,
    'Ctr Avg_Trend': add_result.trend,
    'Seasonal': add_result.seasonal,
    'd(t)': add_result.trend + add_result.resid,
    'Residual': add_result.resid})

print(add_decomposition.head())
print(add_decomposition.tail())
