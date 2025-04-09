# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 13:28:13 2025

@author: Arjoh
"""

#Library Importation
import math
import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, roc_curve, auc
from dmba import regressionSummary, classificationSummary, liftChart, gainsChart

###--------------Import and Format Data
car_df = pd.read_csv(r"C:\Users\Arjoh\Downloads\ToyotaCorolla.csv")

#Select predictor variables (excluding non-numeric/text-based columns)
excludeColumns = ['Price', 'Id', 'Model', 'Fuel_Type', 'Color']
predictors = [col for col in car_df.columns if col not in excludeColumns]
outcome = 'Price'

#Partition data
X = car_df[predictors]
y = car_df[outcome]
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

#Train linear regression model
reg = LinearRegression()
reg.fit(train_X, train_y)

#Evaluate performance
regressionSummary(train_y, reg.predict(train_X))  # Training
regressionSummary(valid_y, reg.predict(valid_X))  # Validation

###--------------Prediction Plots for regression
#Residual Errors
pred_error_train = pd.DataFrame({'residual': train_y - reg.predict(train_X), 'data set': 'training'})
pred_error_valid = pd.DataFrame({'residual': valid_y - reg.predict(valid_X), 'data set': 'validation'})
boxdata_df = pd.concat([pred_error_train, pred_error_valid], ignore_index=True)

#Plot Prediction Errors
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 4))
common = {'bins': 100, 'range': [-6500, 6500]}
pred_error_train.hist(ax=axes[0], **common)
pred_error_valid.hist(ax=axes[1], **common)
boxdata_df.boxplot(ax=axes[2], by='data set')

axes[0].set_title('Training')
axes[1].set_title('Validation')
axes[2].set_ylim(-6500, 6500)
plt.suptitle('Prediction Errors')
plt.subplots_adjust(bottom=0.15, top=0.85, wspace=0.35)
plt.show()

###--------------Cummulative gains and Lift charts
pred_v = pd.Series(reg.predict(valid_X)).sort_values(ascending=False)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
gainsChart(pred_v, ax=axes[0], color='C2').set_title('Cumulative Gains Chart')
liftChart(pred_v, ax=axes[1], labelBars=False).set_ylabel('Lift')
plt.tight_layout()
plt.show()

#Compute Lift
groups = [int(10 * i / len(pred_v)) for i in range(len(pred_v))]  #generate "decile bin indexes"
meanPercentile = pred_v.groupby(groups).mean()      #caluclate mean over "decile bins"
meanResponse = meanPercentile / pred_v.mean()       #calucate ratio vs. the naive model (overall mean)
meanResponse.index = (meanResponse.index + 1) * 10  #change the index to identify the bins by their name
print('Lift based on meanResponse:', meanResponse[10])

random10 = pred_v.cumsum().iloc[-1] / 10  #expected cumulative price based on naive model for 10% sales
cumPred10 = pred_v.cumsum().iloc[57]  #cumulative price based on model for top 10%
print('Expected cumulative price for 10% random sales:', random10)
print('Cumulative price for top 10% sales:', cumPred10)
print('Lift based on gains chart:', cumPred10 / random10)

###--------------Confusion matrix
owner_df = pd.read_csv(r"C:\Users\Arjoh\Downloads\ownerExample.csv")

## Cutoff = 0.5
predicted = ['owner' if p > 0.5 else 'nonowner' for p in owner_df.Probability]
classificationSummary(owner_df.Class, predicted, class_names=['nonowner', 'owner'])

## Cutoff = 0.25
predicted = ['owner' if p > 0.25 else 'nonowner' for p in owner_df.Probability]
classificationSummary(owner_df.Class, predicted, class_names = ['nonowner', 'owner'])

## Cutoff = 0.75
predicted = ['owner' if p > 0.75 else 'nonowner' for p in owner_df.Probability]
classificationSummary(owner_df.Class, predicted, class_names=['nonowner', 'owner'])

###--------------Accuracy/Error Plot vs. Cutoff value
df = pd.read_csv(r"C:\Users\Arjoh\Downloads\liftExample.csv")
accT = [accuracy_score(df.actual, [1 if p > cutoff else 0 for p in df.prob]) for cutoff in [i * 0.1 for i in range(11)]]

plt.plot([i * 0.1 for i in range(11)], accT, '-', label='Accuracy')
plt.plot([i * 0.1 for i in range(11)], [1 - acc for acc in accT], '--', label='Overall error')
plt.ylim([0,1])
plt.xlabel('Cutoff Value')
plt.legend()
plt.show()

###--------------Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(df.actual, df.prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=[5, 5])
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(loc="lower right")
plt.show()

###--------------Cummulative gains and Lift charts liftexample dataset
df = df.sort_values(by='prob', ascending=False)
gainsChart(df.actual, color='darkorange', figsize=(4, 4))
plt.show()

liftChart(df.actual, labelBars=False)
plt.show()
