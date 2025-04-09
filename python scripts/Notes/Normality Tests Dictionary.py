# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:50:08 2024

@author: Arjoh
"""

#Library Importation
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import shapiro, normaltest, anderson
from statsmodels.graphics.gofplots import qqplot

#Import Data
df = pd.read_csv(r"C:\Users\Arjoh\Downloads\Assignment 10 Data.csv")

###--------------Normality Tests Function
def test_normality(df, column_name):
    normality = {}
    
    #Creates a histogram
    plt.hist(df[column_name])
    plt.title(f'Histogram of {column_name}')
    plt.show()
    
    #Creates a QQ plot (a quartile plot)
    qqplot(df[column_name], line='s')
    plt.title(f'QQ Plot of {column_name}')
    plt.show()

    #Statistical Test 1 - Shapiro-Wilk
    stat_sw, p_sw = shapiro(df[column_name])
    normality['Shapiro-Wilk'] = {'stat': stat_sw, 'p-value': p_sw}
    alpha = 0.05
    if p_sw > alpha:
        sw_result = 'Based on Shapiro-Wilks, the sample looks Gaussian (fail to reject H0)'
    else:
        sw_result = 'Based on Shapiro-Wilks, the sample does not look Gaussian (reject H0)'
    normality['Shapiro-Wilk']['result'] = sw_result

    #Statistical Test 2 – D’Agostino’s K^2
    stat_dak, p_dak = normaltest(df[column_name])
    normality['D’Agostino’s K^2'] = {'stat': stat_dak, 'p-value': p_dak}
    alpha = 0.05
    if p_dak > alpha:
        dak_result = 'Based on D’Agostino’s K^2, the sample looks Gaussian (fail to reject H0)'
    else:
        dak_result = 'Based on D’Agostino’s K^2, the sample does not look Gaussian (reject H0)'
    normality['D’Agostino’s K^2']['result'] = dak_result

    #Statistical Test 3 – Anderson-Darling
    result_ad = anderson(df[column_name])
    normality['Anderson-Darling'] = {'stat': result_ad.statistic, 'critical_values': result_ad.critical_values}
    p = 0
    ad_result = []
    for i in range(len(result_ad.critical_values)):
        sl, cv = result_ad.significance_level[i], result_ad.critical_values[i]
        if result_ad.statistic < cv:
            ad_result.append('%.1f: %.3f, Based on AD, the sample looks Gaussian (fail to reject H0)' % (sl,cv))
        else:
            ad_result.append('%.1f: %.3f, Based on AD, the sample does not look Gaussian (reject H0)' % (sl,cv))
    normality['Anderson-Darling']['result'] = ad_result
    
    #Print the results
    print(f"\nNormality Test Results for {column_name}")
    print(f"Shapiro-Wilk: Statistic = {stat_sw:.3f}, p-value = {p_sw:.3f}")
    print(f"  - Result: {normality['Shapiro-Wilk']['result']}")
    print(f"D’Agostino’s K^2: Statistic = {stat_dak:.3f}, p-value = {p_dak:.3f}")
    print(f"  - Result: {normality['D’Agostino’s K^2']['result']}")
    print(f"Anderson-Darling: Statistic = {result_ad.statistic:.3f}")
    print("Critical Values:")
    for res in ad_result:
        print(f"  - {res}")
    print('------------------------------------------------')

    return normality

#Define variables and run normality tests
variables = ['Variable1', 'Variable2', 'Variable3', 'Variable4']
normality_results = {}

for var in variables: normality_results[var] = test_normality(df, var)

#Convert the results (optional)
normalitytests = pd.DataFrame.from_dict(normality_results, orient='index')
normalitytests.to_csv(r'C:\Users\Arjoh\OneDrive\Documents\Python Scripts\Normality Test Results.csv')

