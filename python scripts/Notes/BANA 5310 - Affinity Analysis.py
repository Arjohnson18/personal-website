# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 10:02:40 2025

@author: Arjoh
"""

#Library Importation
import heapq
from collections import defaultdict
import pandas as pd
import matplotlib.pylab as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
import random
from itertools import chain

import warnings
warnings.filterwarnings("ignore")

###--------------Import and Format Data
file_path1 = r"C:\Users\Arjoh\Downloads\Faceplate.csv"
fp_df = pd.read_csv(file_path1)
fp_df.set_index('Transaction', inplace=True)

###--------------Association Rule Mining on the Faceplate Dataset
#The Apriori algorithm: minimum support of 20%
itemsets = apriori(fp_df, min_support=0.2, use_colnames=True) 

#Association Rules: minimum confidence of 50%
rules = association_rules(itemsets, metric='confidence', min_threshold=0.5)

#Sorted by 'lift' and shows top 6 rules
print("Top 6 association rules (sorted by lift):")
print(rules.sort_values(by=['lift'], ascending=False).head(6))

#Drop redundant rules, excluding support and conviction
print("\nTop 6 association rules (excluding support and conviction):")
print(rules.sort_values(by=['lift'], ascending=False)
      .drop(columns=['antecedent support', 'consequent support', 'conviction'])
      .head(6))

#Filter to get rules with single consequents only
print("\nTop 6 association rules with single consequents:")
print(rules[[len(c) == 1 for c in rules.consequents]].sort_values(by=['lift'], ascending=False).head(6))

#Convert data set into a sparse data frame
sparse_df = fp_df.astype(pd.SparseDtype(fp_df.values.dtype, fill_value=0))
print('Density {}'.format(sparse_df.sparse.density))

###--------------Association Rule Mining on the Sparse Faceplate Dataset
#The Apriori algorithm: minimum support of 20%
itemsets = apriori(sparse_df, min_support=0.2, use_colnames=True)

#Association Rules: minimum confidence of 50%
rules = association_rules(itemsets, metric='confidence', min_threshold=0.5)

#Sorted by 'lift' and shows top 6 rules
print("\nTop 6 association rules from sparse dataframe (sorted by lift):")
print(rules.sort_values(by=['lift'], ascending=False).head(6))

###--------------Prepare a synthetic dataset (randomTransactions) 
#This dataset represents transactions as sets of item IDs.
randomTransactions = [{8}, {3,4,8}, {8}, {3,9}, {9}, {1,8}, {6,9}, {3,5,7,9}, {8}, set(), 
                      {1,7,9}, {1,4,5,8,9}, {5,7,9}, {6,7,8}, {3,7,9}, {1,4,9}, {6,7,8}, {8}, set(), {9},
                      {2,5,6,8}, {4,6,9}, {4,9}, {8,9}, {6,8}, {1,6,8}, {5,8}, {4,8,9}, {9}, {8},
                      {1,5,8}, {3,6,9}, {7,9}, {7,8,9}, {3,4,6,8}, {1,4,8}, {4,7,8}, {8,9}, {4,5,7,9}, {2,8,9},
                      {2,5,9}, {1,2,7,9}, {5,8}, {1,7,8}, {8}, {2,7,9}, {4,6,9}, {9}, {9}, {6,7,8}]
print("\nSample of random transactions:")
print(randomTransactions)

#Extract unique items from all transactions and sort them
uniqueItems = sorted(set(chain.from_iterable(randomTransactions)))

#Create a pandas DataFrame to represent the transactions in a one-hot encoded format
randomData = pd.DataFrame(0, index=range(len(randomTransactions)), columns=uniqueItems) #value of 1 indicates the presence of the item in the transaction

#Populate the DataFrame based on the transactions
for row, transaction in enumerate(randomTransactions):
    for item in transaction:
        randomData.loc[row][item] = 1
print("\nOne-hot encoded DataFrame for random transactions:")
print(randomData.head())

###--------------Association Rule Mining on the Synthetic Dataset 
#Apply Apriori algorithm with a minimum support of 2/length of the dataset
itemsets = apriori(randomData, min_support=2/len(randomData), use_colnames=True)

#Association Rules: minimum confidence of 70%
rules = association_rules(itemsets, metric='confidence', min_threshold=0.7)

#Drop redundant rules, excluding support and conviction
print("\nTop 6 association rules from random transactions (excluding support and conviction):")
print(rules.sort_values(by=['lift'], ascending=False)
      .drop(columns=['antecedent support', 'consequent support', 'conviction'])
      .head(6))
print('-' * 50)
#------------------------------------------------------------------------------
###--------------Import and Format Data
file_path2 = r"C:\Users\Arjoh\Downloads\CharlesBookClub.csv"
all_books_df = pd.read_csv(file_path2)

#Define columns to ignore as they are not book purchases (binary incidence matrix)
ignore = ['Seq#', 'ID#', 'Gender', 'M', 'R', 'F', 'FirstPurch', 'Related Purchase',
          'Mcode', 'Rcode', 'Fcode', 'Yes_Florence', 'No_Florence']
count_books = all_books_df.drop(columns=ignore)
count_books[count_books > 0] = 1 #Converts the counts to binary values (1 if purchased, 0 if not)

print("\nBinary incidence matrix for CharlesBookClub dataset:")
print(count_books.head())

#Calculate the frequency of each item
itemFrequency = count_books.sum(axis=0) / len(count_books)

#Plot the item frequencies as a histogram
ax = itemFrequency.plot.bar(color='blue')
plt.ylabel('Item frequency (relative)')
plt.title('Frequency of Purchased Books')
plt.show()

###--------------Association Rule Mining on the CharlesBookClub Dataset
#Apriori algorithm: minimum support of 200/4000 (5%)
itemsets = apriori(count_books, min_support=200/4000, use_colnames=True)

#Association Rules: minimum confidence of 50%
rules = association_rules(itemsets, metric='confidence', min_threshold=0.5)

#Print the total number of generated rules
print('Number of rules generated', len(rules))

#Display 25 rules with highest lift
print("\nTop 25 association rules (sorted by lift) from CharlesBookClub dataset:")
print(rules.sort_values(by=['lift'], ascending=False).head(25))

#Set pandas display options for better readability
pd.set_option('display.precision', 5)
pd.set_option('display.width', 100)

#Print the top 25 rules by lift, excluding support and conviction
print("\nTop 25 association rules (excluding support and conviction):")
print(rules.sort_values(by=['lift'], ascending=False).drop(columns=['antecedent support', 'consequent support', 'conviction']).head(25))
pd.set_option('display.precision', 6) #Reset pandas display options to default

#Filter rules by number of antecedents (maximum 2) and consequents (maximum 1)
rules = rules[[len(c) <= 2 for c in rules.antecedents]]
rules = rules[[len(c) == 1 for c in rules.consequents]]

#Display the top 10 filtered rules by lift
print("\nTop 10 association rules (max 2 antecedents, 1 consequent):")
print(rules.sort_values(by=['lift'], ascending=False).head(10))

###--------------Collaborative Filtering using Surprise Library
#Sample ratings data for movies
ratings = pd.DataFrame([
    [30878, 1, 4], [30878, 5, 1], [30878, 18, 3], [30878, 28, 3], [30878, 30, 4], [30878, 44, 5], 
    [124105, 1, 4], 
    [822109, 1, 5], 
    [823519, 1, 3], [823519, 8, 1], [823519, 17, 4], [823519, 28, 4], [823519, 30, 5], 
    [885013, 1, 4], [885013, 5, 5], 
    [893988, 1, 3], [893988, 30, 4], [893988, 44, 4], 
    [1248029, 1, 3], [1248029, 28, 2], [1248029, 30, 4], [1248029, 48, 3], 
    [1503895, 1, 4], 
    [1842128, 1, 4], [1842128, 30, 3], 
    [2238063, 1, 3], 
], columns=['customerID', 'movieID', 'rating'])

reader = Reader(rating_scale=(1, 5)) #specify the rating scale
data = Dataset.load_from_df(ratings[['customerID', 'movieID', 'rating']], reader) #Load the ratings data into a Surprise Dataset object
trainset = data.build_full_trainset() #Build the full trainset 

#Define similarity options for item-based collaborative filtering using cosine similarity
sim_options = {'name': 'cosine', 'user_based': False} 

#Initialize the KNNBasic model with the specified similarity options
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset) #Train the model on the training data

#Predict the rating for user '823519' and item '30', given a true rating of 4 
pred = algo.predict(str(823519), str(30), r_ui=4, verbose=True)

###--------------Generating Random Rating Data 
#Set a seed for reproducibility of random data
random.seed(0)
nratings = 5000
#DataFrame with random item IDs, user IDs, and ratings
randomData = pd.DataFrame({
    'itemID': [random.randint(0,99) for _ in range(nratings)],
    'userID': [random.randint(0,999) for _ in range(nratings)],
    'rating': [random.randint(1,5) for _ in range(nratings)],
})

###--------------Function to get top N recommendations for each user
def get_top_n(predictions, n=10):
    #First map the predictions to each user.
    byUser = defaultdict(list)
    for p in predictions:
        byUser[p.uid].append(p)
    #For each user, reduce predictions to top-n
    for uid, userPredictions in byUser.items():
        byUser[uid] = heapq.nlargest(n, userPredictions, key=lambda p: p.est)
    return byUser

####--------------Collaborative Filtering on Random Data
#Convert the random rating data into the format required by the surprise package
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(randomData[['userID', 'itemID', 'rating']], reader)

#Data Partitioning: 75% training, 25% validation
trainset, testset = train_test_split(data, test_size=.25, random_state=1)

##User-based filtering
#compute cosine similarity between users 
sim_options = {'name': 'cosine', 'user_based': True}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)

#Than predict ratings for all pairs (u, i) that are NOT in the training set
predictions = algo.test(testset)

#top 4 recommendations for each user
top_n = get_top_n(predictions, n=4)

#Print the recommended items for each user
print()
print('Top-3 recommended items for each user')
for uid, user_ratings in list(top_n.items())[:5]:
    print('User {}'.format(uid))
    for prediction in user_ratings:
        print('  Item {0.iid} ({0.est:.2f})'.format(prediction), end='')
    print()
print()

##Item-based filtering
#compute cosine similarity between users 
sim_options = {'name': 'cosine', 'user_based': False}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)

#Than predict ratings for all pairs (u, i) that are NOT in the training set.
predictions = algo.test(testset)

#top 4 recommendations for each user
top_n = get_top_n(predictions, n=4)

#Print the recommended items for each user
print()
print('Top-3 recommended items for each user')
for uid, user_ratings in list(top_n.items())[:5]:
    print('User {}'.format(uid))
    for prediction in user_ratings:
        print('  Item {0.iid} ({0.est:.2f})'.format(prediction), end='')
    print()

#Build a model using the full dataset
trainset = data.build_full_trainset()
sim_options = {'name': 'cosine', 'user_based': False}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)

#Predict rating for user 383 and item 7
algo.predict(383, 7)