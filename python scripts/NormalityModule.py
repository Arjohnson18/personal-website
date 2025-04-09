# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:06:22 2024

@author: Arjoh
"""

#Library Importation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, normaltest, anderson
from statsmodels.graphics.gofplots import qqplot


    #Statistical Test 1 - Shapiro-Wilk
def shapirotest(series):
    stat_sw, p_sw = shapiro(series)
    alpha = 0.05
    if p_sw > alpha:
        text = "The sample looks Gaussian (fail to reject H0)"
    else:
        text = "The sample does not look Gaussian (reject H0)"
    return f"Shapiro-Wilk: Statistic: {stat_sw:.3f}, p-value: {p_sw:.3f}, Result: {text}"

    #Statistical Test 2 – D’Agostino’s K^2
def dagostinotest(series):
    stat_dak, p_dak = normaltest(series)
    alpha = 0.05
    if p_dak > alpha:
        text = "The sample looks Gaussian (fail to reject H0)"
    else:
        text = "The sample does not look Gaussian (reject H0)"
    return f"D’Agostino’s K^2 Statistic: {stat_dak:.3f}, p-value: {p_dak:.3f}, Result: {text}"

    #Statistical Test 3 – Anderson-Darling
def ADtest(series):
    result = anderson(series)
    cv = result.critical_values[2]  
    stat = result.statistic
    if stat < cv:
        text = "The sample looks Gaussian (fail to reject H0)"
    else:
        text = "The sample does not look Gaussian (reject H0)"
    return f"Anderson-Darling Statistic: {stat:.3f}, Critical Value: {cv:.3f}, Result: {text}"

#Wrapper function to apply all tests on a grouped DataFrame and collect results
def apply_normality_tests(df, group_column):
    normality_results = {}
    grouped = df.groupby(group_column)  # Group by the specified column
    
    #Apply normality tests for each group
    for group_name, group_data in grouped:
        results = {}
        for column in group_data.select_dtypes(include=[np.number]).columns:
            results[column] = {
                'Shapiro-Wilk': shapirotest(group_data[column]),
                "D'Agostino's K^2": dagostinotest(group_data[column]),
                'Anderson-Darling': ADtest(group_data[column])
            }
        normality_results[group_name] = pd.DataFrame(results).T  # Transpose for better readability

    return normality_results





