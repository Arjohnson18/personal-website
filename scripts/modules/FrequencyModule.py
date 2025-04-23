# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 12:12:32 2024

@author: Arjoh
"""

#Library Importation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


###--------------Frequency Tables and Charts
#Creates frequency tables for each variable
def create_freqtable(df, column_name):
    #Calculate class interval 
    range_value = df[column_name].max() - df[column_name].min()
    interval = range_value / math.sqrt(len(df))
    print(f'Interval for {column_name}: {interval}')
    
    #Define custom boundaries
    classes = np.linspace(df[column_name].min(), df[column_name].max(), 10)
    print(f'Classes for {column_name}: {classes}')
    
    #Cuts the data and create a new DataFrame
    df[f'{column_name}_class'] = pd.cut(df[column_name], bins=classes)
    print(df[[column_name, f'{column_name}_class']])
    
    #Creates Frequency Table and add frequencies "Rel_Freq", "Cum_Freq", and "Cum_Rel_Freq"
    FreqTable = df.groupby(f'{column_name}_class').size().reset_index(name='Frequency')
    total_freq = FreqTable['Frequency'].sum()
    FreqTable['Rel_Freq'] = FreqTable['Frequency'] / total_freq
    FreqTable['Cum_Freq'] = np.cumsum(FreqTable['Frequency'])
    FreqTable['Cum_Rel_Freq'] = np.cumsum(FreqTable['Rel_Freq'])
    print(FreqTable)
    
    #Export Results (optional)
    #csv_freqtable = f'C:\\Users\\Arjoh\\OneDrive\\Documents\\Python Scripts\\{column_name} Frequency Table.csv'
    #FreqTable.to_csv(csv_freqtable, index=False)
    
    #Basic Bar Graphs
    plt.style.use('ggplot')
    Count = FreqTable['Frequency']
    Category = FreqTable[f'{column_name}_class'].astype(str)
    ind = np.arange(len(Category))
    width = 0.45
    
    plt.figure(figsize=(10, 6))
    plt.bar(ind, Count, width, label='Classes')
    plt.xticks(ind, Category, rotation=45)
    plt.ylabel("Frequency")
    plt.xlabel("Classes")
    plt.title(f"Frequency Distribution of {column_name}")
    plt.show()

#Helper function to apply the analysis to multiple columns (Optional)
def analyze_freqtable(df, columns):
    for column in columns:create_freqtable(df, column)
