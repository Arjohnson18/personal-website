# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 09:36:49 2025

@author: Arjoh
"""

#Library Importation
import pandas as pd
import matplotlib.pylab as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

###--------------Import and Format Data 
mower_df = pd.read_csv(r"C:\Users\Arjoh\Downloads\RidingMowers.csv")
mower_df['Number'] = mower_df.index + 1     
 
##--------------Data Partitioning: 60% training, 40% validation
trainData, validData = train_test_split(mower_df, test_size=0.4, random_state=26)
newHousehold = pd.DataFrame([{'Income': 60, 'Lot_Size': 20}])    #record to be classifed

###--------------View a scatter plot of the Training data
fig, ax = plt.subplots()
subset = trainData.loc[trainData['Ownership']=='Owner']
ax.scatter(subset.Income, subset.Lot_Size, marker='o', label='Owner', color='C1')
subset = trainData.loc[trainData['Ownership']=='Nonowner']
ax.scatter(subset.Income, subset.Lot_Size, marker='D', label='Nonowner', color='C0')
ax.scatter(newHousehold.Income, newHousehold.Lot_Size, marker='*', label='New household', color='black', s=150)
plt.title('Training Data')
plt.xlabel('Income')  
plt.ylabel('Lot_Size') 
for _, row in trainData.iterrows():
    #print (_)   #if you want to see who _ is (the index)
    ax.annotate(row.Number, (row.Income + 2, row.Lot_Size))  
    #add Row.Number as text against the dot/diamond point in the graph, 
    #at position Income + 2 (slightly to the right of the dot/diamond, Lot_size)
    handles, labels = ax.get_legend_handles_labels()
ax.set_xlim(40, 115)
ax.legend(handles, labels, loc=4)
plt.show()

###--------------Add Validation Data to the scatter plot
def plotDataset(ax, data, showLabel=True, **kwargs):
    subset = data.loc[data['Ownership']=='Owner']
    ax.scatter(subset.Income, subset.Lot_Size, marker='o', label='Owner' if showLabel else None, color='C1', **kwargs)

    subset = data.loc[data['Ownership']=='Nonowner']
    ax.scatter(subset.Income, subset.Lot_Size, marker='D', label='Nonowner' if showLabel else None, color='C0', **kwargs)
    plt.xlabel('Income')  # set x-axis label
    plt.ylabel('Lot_Size')  # set y-axis label
    for _, row in data.iterrows():
         ax.annotate(row.Number, (row.Income + 2, row.Lot_Size))

fig, ax = plt.subplots()
plotDataset(ax, trainData)    # plot training data as set up above in the first plot 
plotDataset(ax, validData, showLabel=False, facecolors='none')   #plot validation data with same settings as above but no fill color on the dots
ax.scatter(newHousehold.Income, newHousehold.Lot_Size, marker='*', label='New household', color='black', s=150)
plt.title('Training & Validation Data')
plt.xlabel('Income')  
plt.ylabel('Lot_Size')  
handles, labels = ax.get_legend_handles_labels()
ax.set_xlim(40, 115)
ax.legend(handles, labels, loc=4)
plt.show()

###--------------Initialize standardized training, validation, and complete data frames
#Use the training data to learn the transformation.
scaler = preprocessing.StandardScaler()
scaler.fit(trainData[['Income', 'Lot_Size']])  # Note the use of an array of column names; compute the mean and std used for standardization

#Transform the full dataset
mowerStd = pd.concat([pd.DataFrame(scaler.transform(mower_df[['Income', 'Lot_Size']]), 
                                     columns=['zIncome', 'zLot_Size']), mower_df[['Ownership', 'Number']]], axis=1)
trainStd = mowerStd.iloc[trainData.index]  #split into trainnorm and validnorm by using the indices previously chosen atr data partitioning, i.e., from trainData, validData
validStd = mowerStd.iloc[validData.index]
newHouseholdStd = pd.DataFrame(scaler.transform(newHousehold), columns=['zIncome', 'zLot_Size'])

#Apply KNN
knn = NearestNeighbors(n_neighbors=3)
knn.fit(trainStd[['zIncome', 'zLot_Size']])
distances, indices = knn.kneighbors(newHouseholdStd)
print(trainStd.iloc[indices[0], :])  # indices is a list of lists, we are only interested in the first element

#Create a data frame with two columns
train_X = trainStd[['zIncome', 'zLot_Size']]
train_y = trainStd['Ownership']
valid_X = validStd[['zIncome', 'zLot_Size']]
valid_y = validStd['Ownership']

#Train a classifier for different values of k
results = []        #initialize list
for k in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=k).fit(train_X, train_y)
    results.append({
        'k': k,
        'accuracy': accuracy_score(valid_y, knn.predict(valid_X))})

#Convert results to a pandas data frame
results = pd.DataFrame(results)


###--------------Retrain with full dataset
mower_X = mowerStd[['zIncome', 'zLot_Size']]
mower_y = mowerStd['Ownership']
knn = KNeighborsClassifier(n_neighbors=4).fit(mower_X, mower_y)
distances, indices = knn.kneighbors(newHouseholdStd)

###--------------Print Options
print('\nPredicted :')
print(knn.predict(newHouseholdStd))
print('Distances:',distances)
print('-' * 50)
print('Indices:', indices)
print('\nRetrained Data:')
print(mowerStd.iloc[indices[0], :])
