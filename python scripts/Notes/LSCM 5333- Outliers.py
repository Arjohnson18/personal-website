# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:46:39 2025

@author: Arjoh
"""

#Library Importation
import scipy.stats as st
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

#Import Data
df = pd.read_excel(r"C:\Users\Arjoh\Downloads\Assignment 2 Car Data.xlsx")
print(df)

#View a scatter plot of the data
pd.plotting.scatter_matrix(df, diagonal= "hist") #try diagonal= "hist" or "kde"

###--------------Regressions
#Regression Version 1 with SciPy
    #Split the data into predictors and outcome variables 
X = df['Weight']
Y = df['MPG']

Reg_results = st.linregress(X,Y)
slope, intercept, r_value, p_value, stderr = st.linregress(X,Y)

#Regression Version 2 with StatsModels
X = df[['Weight']]
X = sm.add_constant(X)
Y = df['MPG']

model = sm.OLS(Y,X).fit()
summary = model.summary()

#Outlier Statistics
Inf = OLSInfluence(model) #Calculate influence and outlier stats
InfSummary = Inf.summary_table() #extract the influence stats table

###--------------Removing Influential Obs and Variables
df = imputed_df.drop([Obs_Number])  #Rerun the model after dropping the outlier
df = imputed_df.drop(['Var_Name'], axis = 1)    #Drop worst predictor and run

###--------------Print options
print(X)
print(Y)
print(Reg_results)
print(summary)
print(model)
print(Inf)
print(InfSummary)
