from node2vec import Node2Vec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans

# Import Zachary's Karate Club graph
G = nx.karate_club_graph()

# Initialize Node2Vec model
n2v = Node2Vec(G, dimensions=2, walk_length=10, num_walks=100, workers=1, weight_key='weight')

# Learn embeddings
model = n2v.fit(window=5, min_count=1, batch_words=4)

# Store the 2D embeddings in the embedding matrix X
X = np.array([model.wv[str(i)] for i in G.nodes()])

# Apply Kmeans to assign each member to one of the two groups
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)

# Getting the true club labels
labels = [G.nodes[i]['club'] for i in G.nodes()]
color_map = {'Mr. Hi': 0, 'Officer': 1}
y = np.array([color_map[label] for label in labels])

# Plotting both the true labels and KMeans clusters
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: True labels
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=100)
for i, txt in enumerate(G.nodes()):
    axes[0].annotate(txt, (X[i, 0], X[i, 1]))
axes[0].set_title("True Club Memberships")

# Plot 2: KMeans clusters
axes[1].scatter(X[:, 0], X[:, 1], c=clusters, cmap='coolwarm', s=100)
for i, txt in enumerate(G.nodes()):
    axes[1].annotate(txt, (X[i, 0], X[i, 1]))
axes[1].set_title("K-Means Clustering on Embeddings")

plt.tight_layout()
plt.show()
