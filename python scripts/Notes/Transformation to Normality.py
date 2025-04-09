# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 13:06:16 2024

@author: Arjoh
"""

#Import Libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy  
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot

#Import Data
df=pd.read_csv(r"C:\Users\Arjoh\Downloads\Assignment 12 DemoData.csv")

#look at the imported data file
print(df)

#As you can see, there are six variables in the dataset - Variable 1-6
#lets pull one out to work with
#Short cut nameing convention-Let Workdataframe be named wdf
wdf =df['Variable1']

#Lets look at some plots of the data - the histogram and qq plot
pyplot.hist(df)
qqplot(wdf, line='s')

#Descriptive statistics
Descriptives = stats.describe(wdf, bias = False, nan_policy = 'omit')
print(Descriptives)

#Test for normality using a new test - based on D'Agostino and Pearson
#See D’Agostino, R. B. (1971), “An omnibus test of normality for moderate and 
#       large sample size”, Biometrika, 58, 341-348
#See D’Agostino, R. and Pearson, E. S. (1973), “Tests for departure from 
#       normality”, Biometrika, 60, 613-622
#H0: wdf comes from a normal distribution; reject if p < 0.05

k2, p = stats.normaltest(wdf)
print("k^2 =", p)
print("p-value =", p)

###--------------Transformation 1: BoxCox - only works if all variable data are positive
Var_T, lmbda = stats.boxcox(wdf)
pyplot.hist(Var_T)
qqplot(Var_T, line = 's')
print (Var_T)
print("Lambda =", lmbda)
k2, p = stats.normaltest(Var_T)
print("k^2 =", k2)
print("p-value =", p)

###--------------Transformation 2: Yeo-Johnson - when some variable data are negative  
Var_T, lmbda = stats.yeojohnson(wdf)
pyplot.hist(Var_T)
qqplot(Var_T, line = 's')
print (Var_T)
print("Lambda =", lmbda)
k2, p = stats.normaltest(Var_T)
print("k^2 =", k2)
print("p-value =", p)

###--------------Transformation 3: Crude Manual (Optional)
##By arcsin (no good for negative values)
ManVar_T = np.arcsin(wdf)

##By log(x/(1-x)) - might get domain errors when near zero
ManVar_T = math.log(wdf/(1-wdf))

##By square root x
ManVar_T = math.sqrt(wdf)

##By log(x), recall the range of x for log is 0 to infinity. Negative values return NaN
ManVar_T = math.log(wdf)

#Test the transformed results
pyplot.hist(ManVar_T)
qqplot(ManVar_T, line = 's')
print (ManVar_T)
print("Lambda =", lmbda)
k2, p = stats.normaltest(ManVar_T)
print("k^2 =", k2)
print("p-value =", p)

###--------------Store results
##Transformed variable (notice Var_T is an array, not a dataframe) 
np.savetxt(r'C:\Users\Arjoh\OneDrive\Documents\Python Scripts\TransformedVariable2.csv', Var_T, delimiter= ",") #Saves to current work directory

##Save original descriptive statistics - but first convert to a dataframe
DescStats = pd.DataFrame([Descriptives], columns=Descriptives._fields)
DescStats.to_csv(r'C:\Users\Arjoh\OneDrive\Documents\Python Scripts\DescriptiveStats2.csv')

##Append to that file the descriptives of the transformed variable
DescStats_Var_T = stats.describe(Var_T, bias = False, nan_policy = 'omit')
DescStats_VarT = pd.DataFrame([DescStats_Var_T], columns=DescStats_Var_T._fields)
DescStats_VarT.to_csv(r'C:\Users\Arjoh\OneDrive\Documents\Python Scripts\DescriptiveStats2.csv', mode = 'a')




