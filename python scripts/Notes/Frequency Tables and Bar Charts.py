# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:09:35 2024

@author: Arjoh
"""

#This program builds frequency distribution tables and bar charts using DataFrames library commands

# Library Importation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Data
df=pd.read_csv(r"C:\Users\Arjoh\Downloads\Assignment 5 and 7 DataSet.csv")
print(df)

#Generate random numbers
np.random.seed(1234)
df1=pd.DataFrame({'normal':np.random.normal(20,5,1000)})
print(df1)

df1['normal'].min()
df1['normal'].max()
df1.max()
df1.std()

###--------------Frequency Tables

#Break 'normal' data from df1 into classes
#   based on 2^k > n, lets use 10 classses
pd.cut(df1['normal'],10)

# Using custom boundaries using class interval
#   i = (max-min)/ k, then i for the 'normal' data is
i= (df1.max() - df1.min()) / 10
print(i)

#Implementing the interval using 
#   np.linspace(start,stop,nr_classes) from the numpy library
#   the easiest way to find the Stop point is (interval)*(k-1)
classes =  np.linspace(0,(36),10)
print(classes)

#Cut the data and create a new DataFrame
df1['normal_class'] = pd.cut(df1['normal'],classes)
print(df1)
FreqTable = df1.groupby('normal_class').count()[['normal']]
FreqTable = FreqTable.rename(columns={'normal':'Frequency'})
print(FreqTable)

#Add a relative frequency (Rel_Freq)
FreqTable.Frequency.sum()       #This sums the entire column
FreqTable['Rel_Freq'] = FreqTable['Frequency'] / FreqTable.Frequency.sum() 
print(FreqTable)

#Add a cumulative frequencies "Cum_Rel_Freq" and "Cum_Freq"
FreqTable['Cum_Rel_Freq'] = np.cumsum(FreqTable['Rel_Freq'])
FreqTable['Cum_Freq'] = np.cumsum(FreqTable['Frequency'])
print(FreqTable)

FreqTable.to_csv(r'C:\Users\Arjoh\OneDrive\Documents\Python Scripts\Frequency Tables.csv')


###--------------Basic Bar Graph
%matplotlib inline  
#a special function that will present the graph in the Plots area outside the window

plt.style.use('ggplot')
Count = FreqTable['Frequency']
Category = FreqTable.index.astype(str)
print(Category)
print(Count)
ind = np.array([x for x, _ in enumerate(Category)])
print(ind)
width = 0.45

#Run the following commands all at once
plt.bar(Category, Count, width, label = 'Classes')
plt.xticks(ind + width/2,Category)
plt.legend(loc='best')
plt.xticks(rotation=45)
plt.ylabel("Frequency")
plt.xlabel("Category")
plt.title("Frequency Distribution of Normal Random Data")
# plt.show()
plt.savefig(r'C:\Users\Arjoh\OneDrive\Documents\Python Scripts\Frequency Distribution of Normal Random Data.png',bbox_inches='tight')










