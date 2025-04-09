# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 17:13:18 2025

@author: Arjoh
"""

#Library Importation
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

###--------------Import Data
df = pd.read_csv(r"C:\Users\Arjoh\Downloads\BostonHousing.csv").rename(columns={'CAT. MEDV': 'CAT_MEDV'})
print(df.head(4), df.describe()) #Display data summary
cereals_df = pd.read_csv(r"C:\Users\Arjoh\Downloads\Cereals.csv")

###--------------Data Summarization
#Compute DescStats and missing values of one variable
desc_stats = df.agg(['mean', 'std', 'min', 'max', 'median']).T
desc_stats['length'] = len(df)
desc_stats['missing_values'] = df.isnull().sum()
print(desc_stats)

#Compute the correlation matrix in b; round results to 2 decimals
print(df.corr().round(2))

#Tabulate CHAS variable counts
print(df.CHAS.value_counts())

#Create RM bins and compute group means
df['RM_bin'] = pd.cut(df.RM, range(0, 10), labels=False)
pivot_table = df.pivot_table(values='MEDV', index='RM_bin', columns='CHAS', aggfunc='mean', margins=True)
print(pivot_table)

###--------------Correlation Analysis
#Crosstab analysis
propTbl = pd.crosstab(df.CAT_MEDV, df.ZN, normalize='columns').round(2)
propTbl.T.plot(kind='bar', stacked=True)
plt.title('Distribution of CAT.MEDV by ZN')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

###--------------Types of Principal Components Analysis
#PCA for two variables
pcs = PCA(n_components=2).fit(cereals_df[['calories', 'rating']])
print(pd.DataFrame(pcs.components_.T, index=['calories', 'rating'], columns=['PC1', 'PC2'])) 
print(pd.DataFrame(pcs.transform(cereals_df[['calories', 'rating']]), columns=['PC1', 'PC2']).head())

#PCA for all variables
b = cereals_df.iloc[:, 3:].dropna()
pcs = PCA().fit(b)
pcsSummary = pd.DataFrame({
    'Standard deviation': np.sqrt(pcs.explained_variance_),
    'Proportion of variance': pcs.explained_variance_ratio_,
    'Cumulative proportion': np.cumsum(pcs.explained_variance_ratio_)
}).T.round(4)
pcsSummary.columns = [f'PC{i+1}' for i in range(len(pcsSummary.columns))]
print(pcsSummary)
print(pd.DataFrame(pcs.components_.T, columns=pcsSummary.columns, index=b.columns).iloc[:, :2])

#Standardized PCA
gn_df = pd.DataFrame(scale(b), columns=b.columns)
pcs = PCA().fit(gn_df)
print(pd.DataFrame(pcs.components_.T, columns=pcsSummary.columns, index=gn_df.columns).iloc[:, :5])
