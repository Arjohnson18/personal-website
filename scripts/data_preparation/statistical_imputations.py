# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:19:42 2025

@author: Arjoh
"""

#Library Importation
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.impute import SimpleImputer
import DescStatsModule as DSM

###--------------Import and Format Data
og_df = pd.read_excel(r"C:\Users\Arjoh\Downloads\Assignment 3 Exercise.xlsx").drop(columns='Index')
#Create a copy of the original dataset for analysis
df_no_missing = og_df.dropna()  # Remove missing values

###--------------Descriptive Statistics
DescStats = DSM.main(og_df)
og_df.info()

#View a scatter plot of the data
pd.plotting.scatter_matrix(og_df, diagonal="hist")

###--------------Statistical Imputation
imputed_df = og_df.copy() #doesn't modify df

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(og_df[['y']])
imputed_df['y'] = imp.transform(og_df[['y']]).ravel()

SIDescStats = DSM.main(imputed_df)

###--------------Regressions with StatsModels
X = og_df[['x']]
X = sm.add_constant(X)
Y = og_df['y']

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
print('\nStatistical Imputation:')
print(SIDescStats)

print(summary)
