# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:16:57 2024

@author: Arjoh
"""

#Library Importation
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from statsmodels.graphics.gofplots import qqplot

#Test data
np.random.seed(1)
#data = np.random.randn(100) + 25
#data = np.random.logistic(loc=10, scale=2, size=1000)

###-----------Import Data
#Code to import real data for testing. Turn on when ready to do work
df = pd.read_pdf(r"C:\Users\Arjoh\Downloads\")

# Extract One column to test. Turned off while developing program
data = df[["column_name_here"]].to_numpy()     #Get variable name from variable explorer

#View the data with a histogram
plt.hist(data)
plt.show()

#view the data with a QQ plot (a quartile plot)
# in the QQ plot, perfectly normal data will have the dots on the diagonal line
qqplot(data, line='s')
plt.show()

###-----------Statistical Test 1 - Shapiro-Wilk
# Shapiro-Wilk test has the null hypothesis H0: data is normally distributed
# Therefore high p-values mean fail to reject (data is normal)
# Small p-values (p < 0.05) mean reject H0 (data is not normal)

stat, p = shapiro(data)
print('SW Statistics=%.3f, p=%.3f'%(stat, p))

# interpret
alpha = 0.05
if p > alpha:
    print('Based on Shapiro-Wilks, the sample looks Gaussian (fail to reject H0)')
else:
    print('Based on Shapiro-Wilks, the sample does not look Gaussian (reject H0)')

###-----------Statistical Test 2 – D’Agostino’s K^2 (from scipy.stats)

stat, p = normaltest(data)
print('DAK Statistics=%.3f, p=%.3f'%(stat, p))

# interpret
alpha = 0.05
if p > alpha:
    print('Based on D’Agostino’s K^2, the sample looks Gaussian (fail to reject H0)')
else:
    print('Based on D’Agostino’s K^2, the sample does not look Gaussian (reject H0)')

###-----------Statistical Test 3 – Anderson-Darling
result = anderson(data)
stat, p = normaltest(data)
print('AD Statistics: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
    sl,cv = result.significance_level[i], result.critical_values[i]
    if result.statistic < result.critical_values[i]:
        print('%.3f: %.3f, Based on AD, the data looks normal (fail to reject H0)' % (sl,cv))
    else:
        print('%.3f: %.3f, Based on AD, the data does not look normal (reject H0)' % (sl,cv))

#print(result)





















