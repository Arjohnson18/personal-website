# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 09:13:06 2024

@author: Arjoh
"""

# Library Importation

import numpy as np
import pandas as pd


# Array of random numbers
    # numpy.random generates random numbers from the normal distribution
a= np.random.randn(4,6)
print(a)

# Use a seed
np.random.seed(1234)
np.random.rand(4,6)

b= np.random.randn(4,6)
print(b)

data = np.random.randint(10,100,size=1000)
print(data)

# convert array to dataframe
df = pd.DataFrame(data, columns = ['Random_Numbers'])
print(df)

#generate random numbers in three columns with mean=15 and stdev=7.7, and 150 obs in three columns
data2 = np.random.normal(loc=15, scale=7.7, size=(150,3))
print(data2)
df2=pd.DataFrame(data2, columns=['Var1', 'Var2','Var3'])
print(df2)

#import csv data into an array
c = np.genfromtxt(r"C:\Users\Arjoh\Downloads\Assignment 5 and 7 DataSet.csv", delimiter=',')
print(c)

#import .csv into DataFrame
df3=pd.read_csv(r"C:\Users\Arjoh\Downloads\Assignment 5 and 7 DataSet.csv")
print(df3)

#call data from the dataFrame

df3.loc[175:]
df3.loc[175:175]
df3.loc[1:1, 'Location']
df3.loc[1,'Location':]

#generate a new dataframe and append (concatentate) to df3
df4= pd.DataFrame([[20,1111,'Stephenville','Pickup',0]], columns= ['Age', 'Profit','Location', 'Vehicle-Type', 'Previous'])
print(df4)

#concatente df4 on to df3
df3 = pd.concat([df3, df4])
print(df3)

#Notice the index
df3 = pd.concat([df3, df4], ignore_index=True)
print(df3)

#concatente df3 on to df4 and create a new DataFrame
df_new=pd.concat([df4, df3], ignore_index=True)
print(df_new)

#Analyzing Data

Mean = df3['Profit'].mean()
Median = df3['Profit'].median()
Min = df3['Profit'].min()
Max = df3['Profit'].max()
Q1 = df3['Profit'].quantile(0.25)
Q2 = df3['Profit'].quantile(0.50)
Q3 = df3['Profit'].quantile(0.75)
Skew = df3['Profit'].skew()
Kurt = df3['Profit'].kurt()
StDev = df3['Profit'].std()
Var = df3['Profit'].var()

List = [Mean,Median, Min, Max, Q1, Q2, Q3, Skew, Kurt, StDev,Var]
ColNames = ['Mean','Median', 'Min', 'Max', 'Quantile1', 'Quantile2', 'Quantile3', 'Skew', 'Kurt', 'StDev','Var']
ProfitAnalysisResults = pd.DataFrame([List], columns=ColNames)
print(ProfitAnalysisResults)

ProfitAnalysisResults2 = pd.DataFrame({'Statistic':List}, index=ColNames)
print(ProfitAnalysisResults2)

# Export Results
# Pythons DataFrame Library has several formats for exporting results including:
    #CSV, Excek, TXT, PDF, SAS, DAT, LATEX, STATA, HTML, SQL, and System clipboard
    #data.to_csv(r'C:\Users\Arjoh\OneDrive\Documents\Python Scripts\data.csv')

ProfitAnalysisResults.to_csv(r'C:\Users\Arjoh\OneDrive\Documents\Python Scripts\ProfitAnalysisResults.csv')
