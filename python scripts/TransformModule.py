# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 18:49:21 2024

@author: Arjoh
"""

import numpy as np
import scipy.stats as stats
import NormalityModule

def BoxCox(df, columns):
    results = {}
    for column in columns:
        VarT_BoxCox, lmbda_BoxCox = stats.boxcox(df[column].dropna())
        SW_VarT_BoxCox = NormalityModule.shapirotest(VarT_BoxCox)

        #Store results
        results[column] = {
            "transformed_data": VarT_BoxCox,
            "lambda": lmbda_BoxCox,
            "shapiro_test": SW_VarT_BoxCox}
    
    return results

def YeoJohnson(df, columns):
    results = {}
    for column in columns:
        VarT_YeoJohnson, lmbda_YeoJohnson = stats.boxcox(df[column].dropna())
        SW_VarT_YeoJohnson = NormalityModule.shapirotest(VarT_YeoJohnson)

        #Store results
        results[column] = {
            "transformed_data": VarT_YeoJohnson,
            "lambda": lmbda_YeoJohnson,
            "shapiro_test": SW_VarT_YeoJohnson}
    
    return results

