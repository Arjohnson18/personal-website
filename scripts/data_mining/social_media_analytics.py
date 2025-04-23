# %%
import collections
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

###--------------Build a dataframe that defines the edges and use to build the graphs
df = pd.DataFrame([
    ("Dave", "Jenny"), ("Peter", "Jenny"), ("John", "Jenny"),
    ("Dave", "Peter"), ("Dave", "John"), ("Peter", "Sam"),
    ("Sam", "Albert"), ("Peter", "John")
], columns=['from', 'to'])

#Create a undirected graph 
G = nx.from_pandas_edgelist(df, 'from', 'to')
nx.draw(G, with_labels=True, node_color='lightblue', node_size=1600)
plt.show()

#Create a directed graph using nx.DiGraph
G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph())
nx.draw(G, with_labels=True, node_color='lightblue', node_size=1600)
plt.show()

# %%
###--------------Code for plotting the drug money laundries network
drug_df = pd.read_csv('drug.csv')
G = nx.from_pandas_edgelist(df, 'Entity', 'Related Entity')
centrality = nx.eigenvector_centrality(G)
nx.draw(G, with_labels=False, node_color='skyblue',
        node_size=[400 * centrality[n] for n in G.nodes()])
plt.show()

# Code for plotting different layouts
G = nx.from_pandas_edgelist(df, 'from', 'to')
plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

plt.subplot(121)
nx.draw_circular(G, with_labels=True, node_color='lightblue', node_size=1600)

plt.subplot(122)
nx.draw_kamada_kawai(G, with_labels=True, node_color='lightblue', node_size=1600)
plt.tight_layout()
plt.show()
# %%
###--------------Adjacency matrix
G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph())
a = nx.to_numpy_array(G)
print(a)

# Compute degree distribution
G = nx.from_pandas_edgelist(df, 'from', 'to')
print(G.degree())

# Print the degree for Peter
print(G.degree['Peter'])

# Compute centrality
print('Centrality: ')
print(nx.closeness_centrality(G))

print('Betweenness: ')
print(nx.betweenness_centrality(G, normalized=False))
print(nx.betweenness_centrality(G))

print('Eigenvector centrality: ')
print(nx.eigenvector_centrality(G, tol=1e-2))
# v = nx.eigenvector_centrality(G, tol=1e-2).values()  # Access the values if need be

nx.draw_kamada_kawai(G, with_labels=True, node_color='lightblue', node_size=1600)
plt.show()

print(nx.betweenness_centrality(G))
print(nx.current_flow_betweenness_centrality(G))  # Variant of betweenness

###--------------Visualize egocentric networks
print(G.nodes)

plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# Print 1st level (degree 1) egocentric network
plt.subplot(121)
G_ego = nx.ego_graph(G, 'Peter')
nx.draw(G_ego, with_labels=True, node_color='lightblue', node_size=1600)

# Print 2nd level (degree 2) egocentric network
plt.subplot(122)
G_ego = nx.ego_graph(G, 'Peter', radius=2)
nx.draw(G_ego, with_labels=True, node_color='lightblue', node_size=1600)

plt.show()

###--------------Compute degree distribution
degreeCount = collections.Counter(d for node, d in G.degree())
degreeDistribution = [0] * (1 + max(degreeCount))
for degree, count in degreeCount.items():
    degreeDistribution[degree] = count
print('Degree distribution:', degreeDistribution)

degreeCount = collections.Counter(d for node, d in G.degree())
print('Degree count:', degreeCount)

print('Network Density:', nx.density(G))
b = [d / sum(degreeDistribution) for d in degreeDistribution]  # Normalized
print(b)

###---------------- Code for Twitter interface -----------------###
import os
from twython import Twython

# credentials = {}
# credentials['CONSUMER_KEY'] = os.environ.get('TWITTER_CONSUMER_KEY', None)
# credentials['CONSUMER_SECRET'] = os.environ.get('TWITTER_CONSUMER_SECRET', None)

# # if not (credentials['CONSUMER_KEY'] is None and credentials['CONSUMER_SECRET'] is None):
# python_tweets = Twython(credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET'])

# # Create our query
# query = {'q': 'text mining', 'result_type': 'recent',
#          'count': 25, 'lang': 'en'}

# recentTweets = python_tweets.search(**query)
# for tweet in recentTweets['statuses'][:2]:
#     print(tweet['text'])
