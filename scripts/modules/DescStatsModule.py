# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:52:00 2024

@author: Arjoh
"""

import pandas as pd

def main(df, column_name=None):
    stats = {}  #Dictionary to store results
    
    if column_name:  #If a column name, group by that column
        defined_groups = df.groupby(column_name)

        for group_name, group_df in defined_groups:
            group_stats = {
                'Mean': group_df.mean(),
                'Median': group_df.median(),
                'Count': group_df.count(),
                'Min': group_df.min(),
                'Max': group_df.max(),
                'StDev': group_df.std(),
                'Variance': group_df.var(),
                'Skewness': group_df.skew(),
                'Kurtosis': group_df.kurt()}
            
            #Convert dictionary to a DataFrame and round values
            stats[group_name] = pd.DataFrame(group_stats).round(3)

    else:  #If no column name, compute stats for all numeric columns
        stats = {
            'Mean': df.mean(),
            'Median': df.median(),
            'Count': df.count(),
            'Min': df.min(),
            'Max': df.max(),
            'StDev': df.std(),
            'Variance': df.var(),
            'Skewness': df.skew(),
            'Kurtosis': df.kurt()}

        #Convert dictionary to a DataFrame and round values
        stats = pd.DataFrame(stats).T.round(3)

    return stats  #Returns a DataFrame if no grouping, else a dictionary of DataFrames
