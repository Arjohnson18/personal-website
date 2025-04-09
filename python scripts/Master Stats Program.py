# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:46:16 2024

@author: Arjoh
"""

#Library Importation
import pandas as pd
import numpy as np
import scipy
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot as plt

import DescStatsModule
import NormalityTest
import Transform2
import goodness_of_fit_module as gof

#Import Data
df=pd.read_csv(r"C:\Users\Arjoh\Downloads\Assignment 12 Data.csv")
variables = ['Variable1', 'Variable2', 'Variable3']

#Let WorkDataFrame be named wdf
#wdf =df['Variable1']

###--------------Descriptive Statistics
DescStats = DescStatsModule.main(df)

###--------------Normality Tests- Small p-values (p < 0.05) mean reject H0 (data is not normal)
#Shapiro-Wilks 
Shapiro_WilksTest = NormalityTest.Shapiro(df) 
#D'Agostino K^2
DAgostinoTest = NormalityTest.DAgostino(df) 
#Anderson-Darling 
Anderson_DarlingTest = NormalityTest.AD(df) 

###--------------Goodness of fit Test
gof_results = gof.run_gof_tests(df, variables)



###--------------Transform Data to Normality
#BoxCox
Var_T, lmbda, SW_TransVar = Transform2.BoxCox(df) #Box-Cox transformation - no neg values
#Yeo-Johnson
Var_T, lmbda, SW_TransVar = Transform2.YeoJohnson(df)#Yeo-Johnson transformation

#Plots of data
plt.hist(df)
qqplot(df, line='s')

plt.hist(Var_T)
qqplot(Var_T, line='s')


###--------------Print options
print(df)
print(DescStats)
print(Shapiro_WilksTest)
print(DAgostinoTest)
print(Anderson_DarlingTest)
print(Var_T)
print(lmbda)
print(SW_TransVar)

###--------------Save Options
type(df)
type(lmbda) 
type(str(lmbda))
type(SW_TransVar)
type(DescStats)
type(Shapiro_WilksTest)

#To save a DataFrame, use OBJname.to_csv
DescStats.to_csv(r'C:\Users\Arjoh\OneDrive\Documents\Python Scripts\DescriptiveStats_Var1.csv', mode='a') 		# mode =’a’ appends

#To save numpy.float64, convert it to a string (str) and use print
print(str(lmbda),file=open(r'C:\Users\Arjoh\OneDrive\Documents\Python Scripts\Lambda.txt','w'))

#To save a list, use print
print(str(Shapiro_WilksTest),file=open(r'C:\Users\Arjoh\OneDrive\Documents\Python Scripts\SW.txt','w'))







