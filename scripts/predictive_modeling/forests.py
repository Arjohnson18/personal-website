# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 12:47:42 2025

@author: Arjoh
"""
#Library Importation
import numpy as np
import pandas as pd

import matplotlib.pylab as plt
from dmba import classificationSummary, regressionSummary

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

#Tree visualization 
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus

###-------------- Import and Format Data
toyotaCorolla_df = pd.read_csv(r"C:\Users\Arjoh\Downloads\ToyotaCorolla.csv").iloc[:1000,:] #selects the first 1000 records for use
toyotaCorolla_df = toyotaCorolla_df.rename(columns={'Age_08_04': 'Age', 'Quarterly_Tax': 'Tax'}) #renames selected columns

#Select Predictors and Outcome Variable
predictors = ['Age', 'KM', 'Fuel_Type', 'HP', 'Met_Color', 'Automatic', 'CC', 
              'Doors', 'Tax', 'Weight']
outcome = 'Price'

#Create dummies, X and y dataframes
X = pd.get_dummies(toyotaCorolla_df[predictors], drop_first=True)
y = toyotaCorolla_df[outcome] #Target

###--------------Training
#Data Partitioning: 60% training, 40% validation
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

###--------------Exhaustive grid search
#Initial search to find optimized tree
param_grid = {
     'max_depth': [5, 10, 15, 20, 25], 
     'min_impurity_decrease': [0, 0.001, 0.005, 0.01], 
     'min_samples_split': [10, 20, 30, 40, 50]}
gridSearch = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_X, train_y)
print('Initial parameters: ', gridSearch.best_params_)

#Fine tuning based on result from initial search
param_grid = {
     'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
     'min_impurity_decrease': [0, 0.001, 0.002, 0.003, 0.005, 0.006, 0.007, 0.008], 
     'min_samples_split': [14, 15, 16, 18, 20, ],}

gridSearch = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, n_jobs=-1) #5-fold cross-validation
gridSearch.fit(train_X, train_y)
print('Improved parameters: ', gridSearch.best_params_)

#Compute prediction tree performance on training set, validation set
regTree = gridSearch.best_estimator_ #Best Regression Tree estimator
TrainTree = regressionSummary(train_y, regTree.predict(train_X))
ValidTree = regressionSummary(valid_y, regTree.predict(valid_X))

print(TrainTree)
print(ValidTree)
print('-' * 50)

###--------------Visualize the tree in python - save graph as a file
dot_data = StringIO()
export_graphviz(regTree, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('regression_tree.png')
Image(graph.create_png())

#predict new record; Note that the parameters you pass to the new record 
#should be according to train_X table because you train the tree using train_X instances
newrec = pd.DataFrame([{'Age': 26, 'KM': 59000, 'HP': 135, 'Met_Color': 1,
                        'Automatic': 1, 'CC': 2100, 'Doors': 4, 'Tax': 200,
                        'Weight': 1200, 'Fuel_Type_Diesel':0,
                        'Fuel_Type_Petrol': 1}])    #record to be classifed

#Compute predicted value (y_hat) and print as $ with 2 decimals
a = regTree.predict(newrec)
print('\nPredicted Price is: ', '${:.2f}'.format(a[0]))


###--------------RANDOM FOREST 
bank_df = pd.read_csv(r"C:\Users\Arjoh\Downloads\UniversalBank.csv")
bank_df.drop(columns=['ID', 'ZIP Code'], inplace=True)

#Select Predictors and Outcome Variable
X = bank_df.drop(columns=['Personal Loan'])
y = bank_df['Personal Loan']
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

#invoque the random forest model and train it on the training dataset
rf = RandomForestClassifier(n_estimators=500, random_state=1)
rf.fit(train_X, train_y)

#Plot features importance plot
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
df = pd.DataFrame({'feature': train_X.columns, 'importance': importances, 'std': std})
df = df.sort_values('importance')
print(df)

#PLot the data
ax = df.plot(kind='barh', xerr='std', x='feature', legend=True)
ax.set_ylabel('')
plt.show()

#Confusion Matrix for validation set
classificationSummary(valid_y, rf.predict(valid_X))

###--------------GRADIENT BOOSTED TREES
boost = GradientBoostingClassifier(n_estimators=500, random_state = 1)
boost.fit(train_X, train_y)
classificationSummary(valid_y, boost.predict(valid_X))