# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:01:20 2025

@author: Arjoh
"""

#Library Importation
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

###--------------Import and Format Data
file_path = r"C:\Users\Arjoh\Downloads\Assignment 7 Data 2.xlsx"
df = pd.read_excel(file_path)
df.head()
data = df['GDP']

#Plot Data
df.plot(x='Month-Year', y='GDP', style='.')
plt.xlabel("Month-Year")
plt.ylabel("Data")
plt.title("Time Series Data")
plt.show()

###--------------Robust model 
#True for LOWESS and False for for LOESS
stl = STL(data, period=12, robust = True).fit()
stl.plot()
plt.xlabel("Month-Year")
plt.title("Robust Model for Data 2")
plt.show()

#Extract output 
stl_decomposition = pd.DataFrame({
    'Observed Y': stl.observed,
    'Ctr Avg_Trend': stl.trend,
    'Seasonal': stl.seasonal,
    'Residual': stl.resid,
    'Weight': stl.weights})

# Show first and last rows
print(stl_decomposition.head())
print(stl_decomposition.tail())
