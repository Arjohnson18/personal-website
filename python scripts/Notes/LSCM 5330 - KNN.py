# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:43:30 2025

@author: Arjoh
"""

#Library Importation
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.impute import KNNImputer
import DescStatsModule as DSM

###--------------Import and Format Data
df = pd.read_excel(r"C:\Users\Arjoh\Downloads\Assignment 3 Exercise.xlsx").drop(columns='Index')

###--------------Descriptive Statistics
DescStats = DSM.main(df)

#View a scatter plot of the data
pd.plotting.scatter_matrix(df, diagonal="hist")

###--------------Nearest Neighbors Imputation
imputed_df = df.copy() 

imp = KNNImputer(n_neighbors=2, weights="uniform")
imputed_df = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
    #includes, the fit, transform, annd columns functions
NNIDescStats = DSM.main(imputed_df)

###--------------Regressions with StatsModels
X = df[['x']]
X = sm.add_constant(X)
Y = df['y']

model = sm.OLS(Y,X).fit()
summary = model.summary()

#Outlier Statistics
Inf = OLSInfluence(model) #Calculate influence and outlier stats
InfSummary = Inf.summary_table() #extract the influence stats table

###--------------Removing Influential Obs and Variables
#df = imputed_df.drop([Obs_Number])  #Rerun the model after dropping the outlier
#df = imputed_df.drop(['Var_Name'], axis = 1)    #Drop worst predictor and run

###--------------Print Options
print('\nOriginal Data:')
print(DescStats)
print('-' * 50)
print('\nNearest Neighbors Imputation:')
print(NNIDescStats)
print('-' * 50)

print(summary)