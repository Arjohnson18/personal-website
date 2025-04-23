# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:32:10 2025

@author: Arjoh
"""

#%matplotlib inline #for tree visualization in iPython/Jupyter notebooks
######THE GRAPHICAL TREE DISPLAY DOES NOT WORK IN PYTHON< ONLY WORKS IN JUPYTER NOTEBOOKS (BASED ON IPYTHON)
#Library Importation
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

import matplotlib.pylab as plt
from dmba import plotDecisionTree, classificationSummary, regressionSummary

#Tree visualization 
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus

###--------------Import and Format Data 
df = pd.read_csv(r"C:\Users\Arjoh\Downloads\ionosphere.csv")

#Select Predictors and Outcome Variable
X = df.iloc[:, 0:33]  #Predictor; Selects columns from index 0 to 33
y = df['CLASS'] #Target

###--------------Training
#Data Partitioning: 60% training, 40% validation
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)

#Train Full Decision Tree Classifier
fullClassTree = DecisionTreeClassifier()
fullClassTree.fit(train_X, train_y)

#Confusion Matrix for Full Tree
classificationSummary(train_y, fullClassTree.predict(train_X))
classificationSummary(valid_y, fullClassTree.predict(valid_X))

#5-fold cross-validation of the full decision tree classifier (accuracy)
treeClassifier = DecisionTreeClassifier()
scores = cross_val_score(treeClassifier, train_X, train_y, cv=5)
    #100% accurate in classifying
    #full-grown tree overfits the training data to perfect accuracy
    
print('Accuracy scores of each fold: ', [f'{acc:.3f}' for acc in scores])
print(f'Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})') #print mean +/- 2 stdevs
print(f'Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})') #print mean +/- 1 stdev

###--------------Parameter Tuning
    #Consists of using an exhaustive grid search to find the combination that leads to the 
    #tree with smallest error (highest accuracy). Then, we use cross-validation 
    #on the training set, and, after settling on the best tree, use that tree 
    #with the validation data to evaluate likely actual performance with new data.
    
#Train Small Decision Tree Classifier to prevent overfitting (stops tree growth)
smallClassTree = DecisionTreeClassifier(max_depth=30, min_samples_split=20, min_impurity_decrease=0.01)
smallClassTree.fit(train_X, train_y)

#Confusion Matrix for Small Tree
classificationSummary(train_y, smallClassTree.predict(train_X))
classificationSummary(valid_y, smallClassTree.predict(valid_X))

###--------------Exhaustive grid search
#Initial search
param_grid = {
    'max_depth': [10, 20, 30, 40], 
    'min_samples_split': [20, 40, 60, 80, 100], 
    'min_impurity_decrease': [0, 0.0005, 0.001, 0.005, 0.01], 
}
gridSearch = GridSearchCV(DecisionTreeClassifier(random_state = 1), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_X, train_y)
print('Initial score: ', gridSearch.best_score_)
print('Initial parameters: ', gridSearch.best_params_)

#Adapt grid based on result from initial search
param_grid = {
    'max_depth': list(range(2, 16)), 
    'min_samples_split': list(range(10, 22)), 
    'min_impurity_decrease': [0.0009, 0.001, 0.0011], 
}
gridSearch = GridSearchCV(DecisionTreeClassifier(random_state=1), param_grid, cv=5, n_jobs=-1)
gridSearch.fit(train_X, train_y)
print('Improved score: ', gridSearch.best_score_)
print('Improved parameters: ', gridSearch.best_params_)

bestClassTree = gridSearch.best_estimator_

###--------------Evaluating performance
#Confusion Matrix for Best Tree
classificationSummary(train_y, bestClassTree.predict(train_X))
classificationSummary(valid_y, bestClassTree.predict(valid_X))

#Plot the Best Tree
plotDecisionTree(bestClassTree, feature_names=train_X.columns)

#Information about the tree
print('Number of nodes', bestClassTree.tree_.node_count)
print('Tree depth', bestClassTree.tree_.max_depth)

estimator = bestClassTree

###--------------Visualize the Best Tree 
dot_data = StringIO()
export_graphviz(bestClassTree, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')
Image(graph.create_png())

#Parse the tree structure
n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold
value = estimator.tree_.value

#Compute Number of Terminal Leaves
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id] = parent_depth + 1

    # If we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

from collections import Counter
nodeClassCounter = Counter()
terminal_leaves = 0
for i in range(n_nodes):
    if is_leaves[i]:
        terminal_leaves = terminal_leaves + 1
        nodeClassCounter.update([np.argmax(value[i][0])])
print()
print('Number of terminal leaves', terminal_leaves)
print(nodeClassCounter)

###--------------Build a new tree and prune the tree based on maximum depth of the tree
#Compare Gini vs Entropy for Various Depths
max_depth = []
acc_gini = []
acc_entropy = []
for i in range(1,30):
     dtree = DecisionTreeClassifier(criterion='gini', max_depth=i)
     dtree.fit(train_X, train_y)
     pred = dtree.predict(valid_X)
     acc_gini.append(accuracy_score(valid_y, pred))
     ####
     dtree = DecisionTreeClassifier(criterion='entropy', max_depth=i)
     dtree.fit(train_X, train_y)
     pred = dtree.predict(valid_X)
     acc_entropy.append(accuracy_score(valid_y, pred))
     ####
     max_depth.append(i)
d = pd.DataFrame({'acc_gini':pd.Series(acc_gini), 
                  'acc_entropy':pd.Series(acc_entropy),
                  'max_depth':pd.Series(max_depth)})

###--------------Visualization
#Plot Accuracy vs. Tree Depth (parameters)
plt.plot('max_depth','acc_gini', data=d, label='gini')
plt.plot('max_depth','acc_entropy', data=d, label='entropy')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.legend()
plt.show()

#Train Full Tree with Entropy Criterion
dtree = DecisionTreeClassifier(criterion='entropy')
dtree.fit(train_X, train_y)
pred = dtree.predict(valid_X)
print('Full Tree Accuracy = ', accuracy_score(valid_y, pred))

#Visualize Full Tree
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('full_tree.png')
Image(graph.create_png())
print('Full Tree depth', dtree.tree_.max_depth)

#Pruned Tree Based on Accuracy (max accuracy: max_depth = 4, criterion = 'entropy')
dtree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
dtree.fit(train_X, train_y)
pred = dtree.predict(valid_X)
print('Pruned Tree Accuracy = ', accuracy_score(valid_y, pred))

#Visualize Pruned Tree
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('pruned_tree.png')
Image(graph.create_png())
print('Pruned Tree depth', dtree.tree_.max_depth)