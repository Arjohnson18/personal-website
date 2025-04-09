# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 12:03:16 2025

@author: Arjoh
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

###-------------- Import and Format Data
df_original = pd.read_excel(r"C:\Users\Arjoh\Downloads\Insert File.xlsx").drop(columns='Index')
df_clean = df_original.dropna()  # Removes missing values

###--------------Regression on Clean Data
X = df_clean[['predictor']]
X = sm.add_constant(X)
Y = df_clean['outcome']

og_model = sm.OLS(outcome, predictor).fit()
og_summary = og_model.summary()

#Outlier Statistics
og_Inf = OLSInfluence(og_model)
og_InfSummary = og_Inf.summary_table()

###--------------Outlier Removal
cooks_d = og_Inf.cooks_distance[0] #Cook’s Distance
hat_diag = og_Inf.hat_matrix_diag #Hat Diagonal (Leverage)
dffits = og_Inf.dffits[0] #DFFITS
student_resid = og_Inf.resid_studentized_external  #External Studentized Residuals

#Mahalanobis Distance Calculation
X_vals = df_clean[['x']].values
X_mean = np.mean(X_vals, axis=0)
X_cov = np.cov(X_vals.T) if X_vals.shape[1] > 1 else np.array([[np.var(X_vals)]])
X_cov_inv = np.linalg.inv(X_cov)
mahalanobis_dist = [mahalanobis(x, X_mean, X_cov_inv) for x in X_vals]

#Save methods to a DataFrame
outliers_df = pd.DataFrame({
    'x': df_clean['x'],
    'y': df_clean['y'],
    'Cook\'s D': cooks_d,
    'Hat Diagonal': hat_diag,
    'DFFITS': dffits,
    'Studentized Residuals': student_resid,
    'Mahalanobis Distance': mahalanobis_dist})

n = len(df_clean) #observations
k = X.shape[1] - 1  #predictors

#Methods
cooks_threshold = 4 / n  #any observation with a value greater than 0.5 is significant
hat_threshold = (2 * (k + 1)) / n 
dffits_threshold = 2 * np.sqrt(k / n)  #if large, the obs is significant
mahalanobis_threshold = chi2.ppf(0.975, df=k)  #any observation with a value greater than 0.975 is an outlier

#Finding outliers
outliers_df['Outlier_Cook'] = outliers_df['Cook\'s D'] > cooks_threshold
outliers_df['Outlier_Hat'] = outliers_df['Hat Diagonal'] > hat_threshold
outliers_df['Outlier_DFFITS'] = abs(outliers_df['DFFITS']) > dffits_threshold
outliers_df['Outlier_Student'] = abs(outliers_df['Studentized Residuals']) > 3
outliers_df['Outlier_Mahalanobis'] = outliers_df['Mahalanobis Distance'] > mahalanobis_threshold

#Outlier conditions
outliers_df['Is_Outlier'] = outliers_df[
    ['Outlier_Cook', 'Outlier_Hat', 'Outlier_DFFITS', 'Outlier_Student', 'Outlier_Mahalanobis']].any(axis=1)

outliers = outliers_df[outliers_df['Is_Outlier']] #Save dropped outliers
#df_no_outliers = df_clean.loc[~outliers_df['Is_Outlier']] #Remove all outliers
#remove selected outliers
df_original = imputed_df.drop([obs]) 
df_original = imputed_df.drop([obs]) 
df_original = imputed_df.drop([obs]) 

###--------------Print Results
#print("\nEach Model's Variance':") #makes sure all models perform accurately
#print(df_imputed_mean['y'].var(), df_imputed_mice['y'].var(), df_imputed_knn['y'].var())

#removed outliers
print("\nOutliers:")
print(outliers[['x', 'y', 'Cook\'s D', 'Studentized Residuals', 'Mahalanobis Distance']])

###--------------Export results
#outliers_df.to_excel(r"C:\Users\Arjoh\Downloads\Outliers_Analysis.xlsx", index=False)
#dropped_outliers.to_excel(r"C:\Users\Arjoh\Downloads\Dropped_Outliers.xlsx", index=False)
