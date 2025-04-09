# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:41:19 2025

@author: Arjoh
"""

#Library Importation
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
import DescStatsModule as DSM

###--------------Import and Format Data
df = pd.read_excel(r"C:\Users\Arjoh\Downloads\Assignment 3 Exercise.xlsx")
df = df.drop(columns= 'Index')

###--------------Descriptive Statistics
DescStats = DSM.main(df)

###--------------Choose One Method
#Method 1: Statistical Imputation
imputed_df1 = df.copy() #doesn't modify df
imp1 = SimpleImputer(missing_values=np.nan, strategy='median')
imp1 = imp1.fit(df[['y']])
imputed_df1['y'] = imp1.transform(df[['y']]).ravel()

SIDescStats = DSM.main(imputed_df1)

#Method 2: Multivariate Imputation by Chained Equations (MICE)
imp2 = IterativeImputer(max_iter=10, verbose=0)
imputed_df2 = pd.DataFrame(imp2.fit_transform(df), columns=df.columns) 
    #includes, the fit, transform, annd columns functions
MICEDescStats = DSM.main(imputed_df2)

#Method 3: Nearest Neighbors Imputation
imp3 = KNNImputer(n_neighbors=2, weights="uniform")
imputed_df3 = pd.DataFrame(imp3.fit_transform(df), columns=df.columns)
    #includes, the fit, transform, annd columns functions
NNIDescStats = DSM.main(imputed_df3)

###--------------Print Options
print('\nOriginal Data:')
print(DescStats)
print('-' * 50)
print('\nStatistical Imputation:')
print(SIDescStats)
print('-' * 50)
print('\nMICE:')
print(MICEDescStats)
print('-' * 50)
print('\nNearest Neighbors Imputation:')
print(NNIDescStats)
print('-' * 50)


