# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 10:35:24 2025

@author: Arjoh
"""

#Library Importation
import pandas as pd  # For handling datasets
from sklearn.model_selection import train_test_split  # For splitting data into training and validation sets
from sklearn.neural_network import MLPClassifier  # For building a neural network classifier
from dmba import classificationSummary  # For generating classification performance summaries

###---------------- Single Hidden Layer (three nodes) -----------------###
###--------------Import and Format Data
df = pd.read_csv(r"C:\Users\Arjoh\Downloads\TinyData.csv")

#Select Predictors and Outcome Variable
predictors = ['Fat', 'Salt']  # Features used for prediction
outcome = 'Acceptance'  # Target variable

# Separate the predictors (X) and target (y)
X = df[predictors]
y = df[outcome]

# Get sorted unique classes from the target variable (used for probability prediction columns)
classes = sorted(y.unique())

###--------------Create a Multi-layer Perceptron (MLP) classifier with:
# - One hidden layer containing 3 neurons
# - Logistic activation function (sigmoid)
# - 'lbfgs' solver (a solver optimized for small datasets)
# - Fixed random state for reproducibility
clf = MLPClassifier(hidden_layer_sizes=(3), 
                    activation='logistic', 
                    solver='lbfgs', 
                    random_state=1)

###--------------Fit and train the model
clf.fit(X, y) 
clf.predict(X)

###--------------Network structure
print('Intercepts')
print(clf.intercepts_)
print('Weights')
print(clf.coefs_)

###--------------Prediction
print("\n---------------- Prediction -----------------")
print(pd.concat([df, pd.DataFrame(clf.predict_proba(X), columns=classes)], axis=1))

###---------------- Single Hidden Layer (Two nodes) -----------------###
###--------------Import and Format Data
accidents_df = pd.read_csv(r"C:\Users\Arjoh\Downloads\accidentsnn.csv")

#Define the input variables
input_vars = ['ALCHL_I', 'PROFIL_I_R', 'VEH_INVL']

#Convert categorical target variables to category type 
accidents_df.SUR_COND = accidents_df.SUR_COND.astype('category')
accidents_df.MAX_SEV_IR = accidents_df.MAX_SEV_IR.astype('category')

#Convert categorical variables into dummy/one-hot encoded variables
processed = pd.get_dummies(accidents_df, columns=['SUR_COND'])
processed = processed.drop(columns=['SUR_COND_9']) #'SUR_COND_9' is excluded to avoid dummy variable trap

#Select Predictors and Outcome Variable
predictors = [c for c in processed.columns if c != outcome]
outcome = 'MAX_SEV_IR'

###--------------Data Partitioning: 60% training, 40% validation
X = processed[predictors]  
y = processed[outcome]  #Target
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

###--------------Train another MLP classifier with:
# - One hidden layer containing 2 neurons
# - Logistic activation function
# - 'lbfgs' solver
# - Fixed random state for reproducibility
clf = MLPClassifier(hidden_layer_sizes=(2), 
                    activation='logistic', 
                    solver='lbfgs',
                    random_state=1)

###--------------Train the model using the training dataset
clf.fit(train_X, train_y.values)

###--------------Evaluate model performance
print("\n---------------- Model Performance -----------------")
classificationSummary(train_y, clf.predict(train_X))
classificationSummary(valid_y, clf.predict(valid_X))
