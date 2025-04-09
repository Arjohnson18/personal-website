# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 16:06:00 2025

@author: Arjoh
"""

#Library Importation
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

import matplotlib.pylab as plt
from dmba import classificationSummary, gainsChart, liftChart
from dmba.metric import AIC_score

###--------------Binary Logistic Regression Model w/ both multiple and simple logistic regression approaches
###Import and Format Data 
df = pd.read_csv(r"C:\Users\Arjoh\Downloads\dummy.csv")

#drop columns and replace " " with "_"
df.drop(columns=['Column1', 'Column2'], errors='ignore', inplace=True)
df.columns = [c.replace(' ', '_') for c in df.columns]

###Convert categorical variables into dummies
df['Education'] = df['Education'].astype('category')
df.Education.cat.rename_categories({1: 'Undergrad', 2: 'Graduate', 3: 'Advanced/Professional'}, inplace=True)
df = pd.get_dummies(df, prefix_sep='_', drop_first=True)

#Variable Selection
y = df['Target']
X = df.drop(columns=['Target'])

#Data Partitioning: 60% training, 40% validation
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

#Fit a logistic regression 
logit_reg = LogisticRegression(penalty='l2', C=1e42, solver='liblinear') #set penalty=l2 and C=1e42 to avoid regularization
logit_reg.fit(train_X, train_y)

#Print results
print('Intercept:', logit_reg.intercept_[0])
a = pd.DataFrame({'Coefficient': logit_reg.coef_[0]}, index=X.columns).transpose()
print(a)
print('AIC:', AIC_score(valid_y, logit_reg.predict(valid_X), df=len(train_X.columns) + 1))

###--------------Simple logistic regression with 1 predictor
#Variable Selection
predictors = ['Predictor']
outcome = 'Target'

X = df[predictors]
y = df[outcome]

#Data Partitioning: 60% training, 40% validation
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

#Fit a logistic regression 
logit_reg = LogisticRegression(penalty="l2", C=1e42, solver='liblinear') #set penalty=l2 and C=1e42 to avoid regularization
logit_reg.fit(train_X, train_y)

#Print results
print('Intercept:', logit_reg.intercept_[0])
print(pd.DataFrame({'Coefficient': logit_reg.coef_[0]}, index=X.columns).transpose())
print('AIC', AIC_score(valid_y, logit_reg.predict(valid_X), df = len(train_X.columns) + 1))

logit_reg_pred = logit_reg.predict(valid_X)
logit_reg_proba = logit_reg.predict_proba(valid_X)
logit_result = pd.DataFrame({'actual': valid_y, 
                              'p(0)': [p[0] for p in logit_reg_proba],
                              'p(1)': [p[1] for p in logit_reg_proba],
                              'predicted': logit_reg_pred })

#Display four different predictor cases
interestingCases = [2764, 932, 2721, 702]
print(logit_result.loc[interestingCases])

###Confusion matrix for training, validation sets
classificationSummary(train_y, logit_reg.predict(train_X))
classificationSummary(valid_y, logit_reg.predict(valid_X))

#Predict probabilities and class for Predictor = 99 (for a new (test) record)
newrec = pd.DataFrame([{'Predictor': 99}])    #record to be classifed
logit_test_proba = logit_reg.predict_proba(newrec)
logit_test_class = logit_reg.predict(newrec)

print(logit_test_proba[0])
print(logit_test_class[0])

#Create plots and graphs of the data
df = logit_result.sort_values(by=['p(1)'], ascending=False)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
gainsChart(df.actual, color = 'blue', ax=axes[0])
liftChart(df['p(1)'], title=False, ax=axes[1])
plt.tight_layout()
plt.show()

###--------------Multivariate Logistic Regression
#Import data:
delays_df = pd.read_csv('FlightDelays.csv')

#Create a Predictor Variable
delays_df['isDelayed'] = [1 if status == 'delayed' else 0 for status in delays_df['Flight Status']]

#Convert Variables to Categorical
delays_df.DAY_WEEK = delays_df.DAY_WEEK.astype('category')


#Create Hourly Bins for Departure Time
delays_df.CRS_DEP_TIME = [round(t / 100) for t in delays_df.CRS_DEP_TIME]
delays_df.CRS_DEP_TIME = delays_df.CRS_DEP_TIME.astype('category')	#time is now hourly, rounded down

#Select Predictors and Outcome Variable
predictors = ['DAY_WEEK', 'CRS_DEP_TIME', 'ORIGIN', 'DEST', 'CARRIER', 'Weather']
outcome = 'isDelayed'

#Convert Categorical Variables into Dummy Variables
X = pd.get_dummies(delays_df[predictors], drop_first=True)
y = delays_df[outcome]
classes = ['ontime', 'delayed']

#Split Data into Training and Validation Sets (60% training, 40% validation)
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

#Train Logistic Regression Model
logit_full = LogisticRegression(penalty="l2", C=1e42, solver='liblinear')
	#L2 is a ridge regression
	#C=1e42 means little regularization
	#liblinear solver is good for small datasets
logit_full.fit(train_X, train_y)

#Display Model Parameters
print('intercept ', logit_full.intercept_[0])
b = pd.DataFrame({'coeff': logit_full.coef_[0]}, index=X.columns).transpose()
print(b)

#Compute AIC Score
print('AIC', AIC_score(valid_y, logit_full.predict(valid_X), df=len(train_X.columns) + 1))

#Predict Probabilities and Results
logit_reg_pred = logit_full.predict_proba(valid_X)
full_result = pd.DataFrame({'actual': valid_y, 
                            'p(0)': [p[0] for p in logit_reg_pred],
                            'p(1)': [p[1] for p in logit_reg_pred],
                            'predicted': logit_full.predict(valid_X)})
full_result = full_result.sort_values(by=['p(1)'], ascending=False)

#Generate Confusion Matrix
classificationSummary(full_result.actual, full_result.predicted, class_names=classes)

#Display Gains Chart
gainsChart(full_result.actual, color = "Blue", figsize=[5, 5])
plt.show()

###--------------Binary, Multivariate, L1-regularized Logistic Regression Model
#Import data:
delays_df = pd.read_csv('FlightDelays.csv')

#Create a Target Variable
delays_df['isDelayed'] = [1 if status == 'delayed' else 0 
                          for status in delays_df['Flight Status']]

#Process Departure Time
delays_df['CRS_DEP_TIME'] = [round(t / 100) for t in delays_df['CRS_DEP_TIME']]
	#groups times into meaningful categories

#Create a Reduced Dataset 
delays_red_df = pd.DataFrame({
    'Sun_Mon' : [1 if d in (1, 7) else 0 for d in delays_df.DAY_WEEK],
    'Weather' : delays_df.Weather,
    'CARRIER_CO_MQ_DH_RU' : [1 if d in ("CO", "MQ", "DH", "RU") else 0 
                              for d in delays_df.CARRIER],
    'MORNING' : [1 if d in (6, 7, 8, 9) else 0 for d in delays_df.CRS_DEP_TIME],
    'NOON' : [1 if d in (10, 11, 12, 13) else 0 for d in delays_df.CRS_DEP_TIME],
    'AFTER2P' : [1 if d in (14, 15, 16, 17, 18) else 0 for d in delays_df.CRS_DEP_TIME],
    'EVENING' : [1 if d in (19, 20) else 0 for d in delays_df.CRS_DEP_TIME],
    'isDelayed' : [1 if status == 'delayed' else 0 for status in delays_df['Flight Status']],})

#Prepare Data for Logistic Regression
X = delays_red_df.drop(columns=['isDelayed'])  #Features (independent variables)
y = delays_red_df['isDelayed']  #Binary Target variable
classes = ['ontime', 'delayed']

#Split Data into Training & Validation Sets (60% training, 40% validation)
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

#Train Logistic Regression Model with L1 Regularization
logit_red = LogisticRegressionCV(penalty="l1", solver='liblinear', cv=5)  # 5-fold cross-validation
logit_red.fit(train_X, train_y)
	#L1 regularization (Lasso) to select important features
	#5-fold cross-validation helps optimize the model

#Display Model Parameters
pd.set_option('display.width', 100)
print('regularization', logit_red.C_)
print('intercept ', logit_red.intercept_[0])
print(pd.DataFrame({'coeff': logit_red.coef_[0]}, index=X.columns).transpose())
pd.reset_option('display.width')
	#Regularization parameter (C_): Controls model complexity
	#Intercept (b0): The baseline probability of being delayed
	#Coefficients (bi): Impact of each predictor on delay probability

#Compute AIC Score
print('AIC', AIC_score(valid_y, logit_red.predict(valid_X), df=len(train_X.columns) + 1))

#Confusion Matrix (Performance Evaluation)
classificationSummary(valid_y, logit_red.predict(valid_X), class_names=classes)
	#True Positives (TP): Correctly predicted delays
	#False Positives (FP): Predicted delays but were actually on time
	#True Negatives (TN): Correctly predicted on-time flights
	#False Negatives (FN): Predicted on-time but were actually delayed

#Compute Predicted Probabilities
logit_reg_proba = logit_red.predict_proba(valid_X)
red_result = pd.DataFrame({'actual': valid_y, 
                            'p(0)': [p[0] for p in logit_reg_proba],
                            'p(1)': [p[1] for p in logit_reg_proba],
                            'predicted': logit_red.predict(valid_X),})
red_result = red_result.sort_values(by=['p(1)'], ascending=False)

#Compare Full & Reduced Model Performance
ax = gainsChart(full_result.actual, label='Full model', color='C1', figsize=[5, 5])
ax = gainsChart(red_result.actual, label='Reduced model', color='C0', ax=ax)
ax.legend()
plt.show()

###--------------Binary Logistic Regression using GLM (Generalized Linear Model)
#same initial preprocessing
bank_df = pd.read_csv('UniversalBank.csv')

#drop columns and replace " " with "_"
bank_df.drop(columns=['ID', 'ZIP Code'], inplace=True)
bank_df.columns = [c.replace(' ', '_') for c in bank_df.columns]

#Convert numerical values (1, 2, 3) to categorial variables
bank_df['Education'] = bank_df['Education'].astype('category')
new_categories = {1: 'Undergrad', 2: 'Graduate', 3: 'Advanced/Professional'}
bank_df.Education.cat.rename_categories(new_categories, inplace=True)

#Converts categorical variables into dummy variables 
bank_df = pd.get_dummies(bank_df, prefix_sep='_', drop_first=True)
    #drops first column to avoid multicollinearity
    
#Add constant column (an intercept term)
bank_df = sm.add_constant(bank_df, prepend=True)

#Variable Selection
y = bank_df['Personal_Loan']    #(Binary: 1 = Accepted, 0 = Rejected)
X = bank_df.drop(columns=['Personal_Loan']) #all remaining columns

#Split Data into Training & Validation Sets (60% training, 40% validation)
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

#Fit a logistic regression
logit_reg = sm.GLM(train_y, train_X, family=sm.families.Binomial())
logit_result = logit_reg.fit()
    #Specifies a Binomial family (used for binary logistic regression)
    #Fits the logistic regression model to the training data
    
print(logit_result.summary())

###--------------Multinomial Logistic Regression (Second Model)
#Code for class with more than 2 levels
data = pd.read_csv('accidentsFull.csv')
outcome = 'MAX_SEV_IR'
predictors = ['ALCHL_I', 'WEATHER_R']

y = data[outcome]
X = data[predictors]
train_X, train_y = X, y
classes = sorted(y.unique())

print('Nominal logistic regression')
logit = LogisticRegression(penalty="l2", solver='lbfgs', C=1e24, multi_class='multinomial')
logit.fit(X, y)
    #penalty="l2": Applies L2 regularization with C=1e24 (effectively removing regularization)
    #solver='lbfgs': Optimization algorithm for multinomial problems
    #multi_class='multinomial': Specifies multinomial regression instead of binary
    
print('Intercept', logit.intercept_[0])
print('Coefficients', logit.coef_[0])
print('Classes', logit.classes_)

probs = logit.predict_proba(X)
results = pd.DataFrame({
    'actual': y, 'predicted': logit.predict(X),
    'P(0)': [p[0] for p in probs],
    'P(1)': [p[1] for p in probs],
    'P(2)': [p[2] for p in probs],})

print(results.head())

#Evaluate Regression
classificationSummary(y, results.predicted, class_names=classes)
