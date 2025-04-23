# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:10:56 2025

@author: Arjoh
"""

#Library Importation
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence


def OLS(df, group_column):
    ###--------------Ordinary Least Squares
    #View a scatter plot of the data
    pd.plotting.scatter_matrix(df, diagonal= "hist") #try diagonal= "hist" or "kde"
    
    #Using StatsModels
    df = pd.read_csv(r"C:\Users\Arjoh\Downloads\Assignment 2 DemoData.csv")
    X = df[['group_column']] #place x-data in a unique object
    X = sm.add_constant(X)
    Y = df['group_column'] #place y-data in a unique object
    
    model = sm.OLS(Y, X).fit()
    print_model = model.summary()
    
    #Outlier Statistics
    Inf = OLSInfluence(model) #Calculate influence and outlier stats
    InfSummary = Inf.summary_table() #extract the influence stats table

    #Print results
    print(print_model)
    print(model)
    print(Inf)
    print(InfSummary)
    
def plot_scatter_matrix(df):
    pd.plotting.scatter_matrix(df, diagonal="hist")  #Displays a scatter matrix plot of the dataframe

def run_ols(df, predictor, response):   #Runs an OLS regression and returns the model summary and influence statistics
    X = df[[predictor]]
    X = sm.add_constant(X)
    Y = df[response]
    
    model = sm.OLS(Y, X).fit()
    summary = model.summary()
    
    #Outlier Statistics
    inf = OLSInfluence(model)
    inf_summary = inf.summary_table()
    
    return summary, inf_summary, model


