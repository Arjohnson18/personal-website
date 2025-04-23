# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 21:44:18 2025

@author: Arjoh
"""

#Library Importation
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


###--------------Import Data
housing_df = pd.read_csv(r"C:\Users\Arjoh\Downloads\WestRoxbury.csv")
print(housing_df)

a = housing_df.shape #find dimension of data frame
print(housing_df.head()) #show the 1st five rows
print(housing_df.tail)
print(housing_df)  #show all the data

#Rename columns: replace spaces with '_' 
housing_df = housing_df.rename (columns={'TOTAL VALUE ': 'TOTAL_VALUE'})  # explicit
print(housing_df)
housing_df.columns = [s.strip().replace(' ', '_') for s in housing_df.columns] # all columns

#Show first four rows of the data
print(housing_df.loc[0:3])    # loc[a:b] gives rows a to b, inclusive 
print(housing_df.iloc[0:4])   #iloc gives rows a to b-1

#Show the fifth row of the first 10 columns
print(housing_df.head)    #this helps compare with what follows
print(housing_df.iloc[4][0:10])
print(housing_df.iloc[4, 0:10])
print(housing_df.iloc[4:5, 0:10])  # use a slice to return a data frame

#Use pd.concat to combine non-consecutive columns into a new data frame. Axis 
#argument specifies dimension along which concatenation happens, 0=rows, 1=columns.
b = pd.concat([housing_df.iloc[4:6,0:2], housing_df.iloc[4:6,4:6]], axis=1)
print(b)
c = pd.concat([housing_df.iloc[0:2,4:6], housing_df.iloc[4:6,4:6]], axis=0)
print(c)

#To specify a full column, use:
print(housing_df.iloc[:,0:1])
print(housing_df.TOTAL_VALUE)
print(housing_df['TOTAL_VALUE'][0:10])  # show the first 10 rows of the first column

###--------------Descriptive statistics
print('Number of rows ', len(housing_df['TOTAL_VALUE'])) # show length of first column
print('Mean of TOTAL_VALUE ', housing_df['TOTAL_VALUE'].mean()) # show mean of column
print(housing_df.describe()) # show summary statistics for each column

#random sample of 20 observations
print(housing_df.sample(20))
print(housing_df.sample(20, random_state = 1).ROOMS) #show only the Rooms column 


#select a sample of 20 and within it, oversample the houses with over 10 rooms
weights_a = [0.9 if rooms > 10 else 0.01 for rooms in housing_df.ROOMS]
print(housing_df.sample(20, weights=weights_a))
print(housing_df.sample(20, weights=weights_a, random_state = 1).ROOMS)

print(housing_df.columns)  #print a list of variables (columns = variables; lines = observations/instances/cases)
# REMODEL needs to be converted to a categorical variable
print(housing_df.REMODEL.dtype)  # Check type of a variable
housing_df.REMODEL = housing_df.REMODEL.astype('category')
print(housing_df.REMODEL.cat.categories)  # Show the categories
print(housing_df.REMODEL.dtype)  # Check type of converted variable
d = housing_df.REMODEL.dtype  # for comparison: print dtype vs. variable d
print(d.name) # for comparison: print dtype vs. variable d

###--------------CREATE DUMMY VARIABLES for REMODEL categories 
#use drop_first=True to drop the first dummy variable
housing_df = pd.get_dummies(housing_df, prefix_sep='_', drop_first=True) #note that no column is specified i.e., all categorical variables will be converted to dummies here
print(housing_df.columns)
print(housing_df.loc[:,'REMODEL_Old':  'REMODEL_Recent'].head(5)) 

#Missing Data Removal and Imputation procedures; we first convert a few entries for  bedrooms to NA's. 
#Then we impute these missing values using the median of the remaining values.
#generate a sample of "pretend" missing data (bc. the original excel fie has no missing data)

missingRows = housing_df.sample(10).index        # notice: no random seed was used
housing_df.loc[missingRows, 'BEDROOMS'] = np.nan
print(housing_df.loc[missingRows, 'BEDROOMS'])
print('Number of (remaining) rows with valid BEDROOMS values after setting to NAN: ', housing_df['BEDROOMS'].count())

#remove rows with missing values 
reduced_df = housing_df.dropna()
print('Number of rows after removing rows with missing values: ', len(reduced_df))

# replace the missing values using the median of the remaining values (the original values in fact)
medianBedrooms = reduced_df['BEDROOMS'].median()  # the book should have used reduced_df here
housing_df.BEDROOMS = housing_df.BEDROOMS.fillna(value=medianBedrooms)
print('Number of rows with valid BEDROOMS values after filling NA values: ', housing_df['BEDROOMS'].count())
print(housing_df.loc[missingRows, 'BEDROOMS'])

###--------------Data Standardization
df = housing_df.copy() 

# using pandas:
std_df = (housing_df - housing_df.mean()) / housing_df.std()
print (std_df.head)

#using scikit-learn: produces a slightly larger count
scaler = StandardScaler()
# the result of the transformation is a numpy array, we convert it into a dataframe
std_dd = pd.DataFrame(scaler.fit_transform(housing_df), index=housing_df.index, columns=housing_df.columns)
print (std_dd.head)

###--------------Data Partitioning

#set random_state for reproducibility
#training (60%) and validation (40%)
trainData, validData = train_test_split(housing_df, test_size=0.40, random_state=1)
print(trainData.head)
print (validData.head)
# produces Training: 3481  Validation: 2321

#training (50%), validation (30%), and test (20%)
trainData, temp = train_test_split(housing_df, test_size=0.5, random_state=1)
#now split temp into validation and test
validData, testData = train_test_split(temp, test_size=0.4, random_state=1)
print(trainData.head)
print (validData.head)
print (testData.head)
#produces  Training:  2901  Validation:  1740   Test:  1161






