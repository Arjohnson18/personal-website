# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:48:06 2025

@author: Arjoh
"""

#Library Importation
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

###--------------Import and Format Data 
df = pd.read_excel(r"C:\Users\Arjoh\Downloads\Assignment 5 Cargo 2000 Data.xlsx")
#df.columns = [c.replace(' ', '_') for c in df.columns]

#Define Variables
x = df[['Planned time']]
y = df['Actual time']

###--------------Fit a logistic model
#Check correlation
correlation = df.corr().loc["Planned time", "Actual time"]

#Fit a linear regression model
X = df[['Planned time']]
y = df['Actual time']

model = LinearRegression()
model.fit(X, y)

###--------------Scatter Plot
plt.figure(figsize=(8, 6))  # Set figure size before plotting
plt.scatter(x, y, alpha=0.7)
plt.xlabel("Planned Time")
plt.ylabel("Actual Time")
plt.title("Scatter Plot of Planned Time vs. Actual Time")
plt.show()

###--------------Predicting Values with the original data set
#When x=200 and x=20,500
x_values = np.array([200, 20500]).reshape(-1, 1)
y_hat = model.predict(x_values)

#OLS regression and prediction
x_ols = sm.add_constant(X)
ols_model = sm.OLS(y, x_ols).fit()
predictions = ols_model.get_prediction(sm.add_constant(x_values))

#Build 95% confidence intervals 
summary_95 = predictions.summary_frame(alpha=0.05)
lower_95 = summary_95["obs_ci_lower"]
upper_95 = summary_95["obs_ci_upper"]
moe_95 = (upper_95 - lower_95) / 2 

#Build 99% confidence intervals 
summary_99 = predictions.summary_frame(alpha=0.01)  
lower_99 = summary_99["obs_ci_lower"]
upper_99 = summary_99["obs_ci_upper"]
moe_99 = (upper_99 - lower_99) / 2  

#Compare Margins of Error
change_in_margin_200 = moe_99[0] - moe_95[0]
change_in_margin_20500 = moe_99[1] - moe_95[1]
average_moe = (change_in_margin_200+change_in_margin_20500)/2

###--------------Outlier Identification
#Compute residuals
residuals = y - ols_model.fittedvalues

#Z-scores
std_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

#Scatter plot of standardized residuals
plt.figure(figsize=(8, 6))
plt.scatter(ols_model.fittedvalues, std_residuals, alpha=0.7)
plt.axhline(0, color="red", linestyle="--", label="Zero Residual Line")
plt.axhline(4, color="blue", linestyle=":", label="Z = 4")
plt.axhline(-4, color="blue", linestyle=":")
plt.xlabel("Fitted Values (Predicted)")
plt.ylabel("Standardized Residuals (Z-score)")
plt.title("Scatter Plot of Standardized Residuals")
plt.legend()
plt.show()

#Number of significant residuals 
outliers_z4 = np.sum(std_residuals > 4)
outliers_z3 = np.sum(std_residuals > 3)

print(f"\nNumber of outliers with Z > 4.0: {outliers_z4}") #highly extreme
print(f"Number of outliers with Z > 3.0: {outliers_z3}")    #significant, but not extreme

###--------------Outlier Removal
#Calculate influence statistics
new_model = OLSInfluence(ols_model)

#Identify outliers 
outliers_z4_list = np.where(np.abs(std_residuals) > 4)[0]
num_outliers_z4 = len(outliers_z4_list)
print(f"\nNumber of outliers with |Z| > 4.0: {num_outliers_z4}")

outliers_z3_list = np.where(np.abs(std_residuals) > 3)[0]
num_outliers_z3 = len(outliers_z3_list)
print(f"Number of outliers with |Z| > 3.0: {num_outliers_z3}")

#Remove outliers abased on std_residuals
excluded_outliers = np.abs(std_residuals) <= 4

#Data without outliers
df_cleaned = df[excluded_outliers]

#Percentage of data deleted
percentage_deleted = (1 - len(df_cleaned) / len(df)) * 100
print(f"Percentage of data deleted: {percentage_deleted:.2f}%")

###--------------Predicting Values with the cleaned data set
#Refit the model using the cleaned data set
cleaned_X = df_cleaned[['Planned time']]
cleaned_y = df_cleaned['Actual time']

cleaned_model = LinearRegression()
cleaned_model.fit(cleaned_X, cleaned_y)

#Predicting values using the cleaned model
cleaned_y_hat = cleaned_model.predict(x_values)

#OLS regression on predicted values
cleaned_x_ols = sm.add_constant(cleaned_X)
cleaned_ols_model = sm.OLS(cleaned_y, cleaned_x_ols).fit()

#Extract standard error of predictions 
cleaned_pred = cleaned_ols_model.get_prediction(sm.add_constant(x_values))

#Build 95% confidence intervals
summary_95c = cleaned_pred.summary_frame(alpha=0.05)
lower_95c = summary_95c["obs_ci_lower"]
upper_95c = summary_95c["obs_ci_upper"]
moe_95c = (upper_95c - lower_95c) / 2

###--------------Predicting Values with the cleaned data set and no constant
#Refit the model using the cleaned data set and no constant
no_intercept_model = LinearRegression(fit_intercept=False)
no_intercept_model.fit(cleaned_X, cleaned_y)

#Predicting values using the no constant model
y_hat_no_intercept = no_intercept_model.predict(x_values)

#Create OLS model without a constant
no_constant_ols = sm.OLS(cleaned_y, cleaned_X).fit()    

#Extract standard error of predictions 
no_constant_pred = no_constant_ols.get_prediction(x_values)

#Build 95% confidence intervals
summary_95nc = no_constant_pred.summary_frame(alpha=0.05)
lower_95nc = summary_95nc["obs_ci_lower"]
upper_95nc = summary_95nc["obs_ci_upper"]
moe_95nc = (upper_95nc - lower_95nc) / 2

###--------------Print all Results
print(f"Correlation coefficient: {correlation:.3f}")

#Model parameters
print(f"\nIntercept: {model.intercept_:.3f}")
print(f"Slope: {model.coef_[0]:.3f}")

#ModelResults
print('-'*50)
print('Model Results')
print(f"Original Model: Predicted values: {y_hat[0]:.3f}, {y_hat[1]:.3f}")
print(f"Original Model: 95% CI: ({lower_95[0]:.3f}, {upper_95[0]:.3f}), ({lower_95[1]:.3f}, {upper_95[1]:.3f})")
print(f"Original Model: MoE: {moe_95[0]:.3f}, {moe_95[1]:.3f}")

print(f"\nCleaned Model: Predicted values: {cleaned_y_hat[0]:.3f}, {cleaned_y_hat[1]:.3f}")
print(f"Cleaned Model: 95% CI: ({lower_95c[0]:.3f}, {upper_95c[0]:.3f}), ({lower_95c[1]:.3f}, {upper_95c[1]:.3f})")
print(f"Cleaned Model: MoE: {moe_95c[0]:.3f}, {moe_95c[1]:.3f}")

print(f"\nNo Constant Model: Predicted values: {y_hat_no_intercept[0]:.3f}, {y_hat_no_intercept[1]:.3f}")
print(f"No Constant Model: 95% CI: ({lower_95nc[0]:.3f}, {upper_95nc[0]:.3f}), ({lower_95nc[1]:.3f}, {upper_95nc[1]:.3f})")
print(f"No Constant Model: MoE: {moe_95nc[0]:.3f}, {moe_95nc[1]:.3f}")

#Compare Models
print('-'*50)
print('Comparisons between models:')

#Comparison between Original and Cleaned Model
if moe_95[0] > moe_95c[0] and moe_95[1] > moe_95c[1]:
    print("For both predicted values, the original model performs worse than the cleaned model.")
elif moe_95[0] < moe_95c[0] and moe_95[1] < moe_95c[1]:
    print("For both predicted values, the original model performs better than the cleaned model.")
else: 
    print("For both predicted values, the original model performs the same as the cleaned model.")

#Comparison between Cleaned and No Constant Model   
if moe_95c[0] > moe_95nc[0] and moe_95c[1] > moe_95nc[1]:
    print("For both predicted values, the cleaned model performs worse than the no constant model.")
elif moe_95c[0] < moe_95nc[0] and moe_95c[1] < moe_95nc[1]:
    print("For both predicted values, the cleaned model performs better than the no constant model.")
else: 
    print("For both predicted values, the cleaned model performs the same as the no constant model.")
    
#Comparison between Original and No Constant Model 
if moe_95[0] > moe_95nc[0] and moe_95[1] > moe_95nc[1]:
    print("For both predicted values, the original model performs worse than the no constant model.")
elif moe_95[0] < moe_95nc[0] and moe_95[1] < moe_95nc[1]:
    print("For both predicted values, the original model performs better than the no constant model.")
else: 
    print("For both predicted values, the original model performs the same as the no constant model.")
