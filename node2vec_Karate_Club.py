from node2vec import Node2Vec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

G = nx.karate_club_graph()

# Initialize Node2Vec model
n2v = Node2Vec(G, dimensions=2, walk_length=10, num_walks=100, workers=1, weight_key='weight')

# Learn embeddings
model = n2v.fit(window=5, min_count=1, batch_words=4)

X = np.array([model.wv[str(i)] for i in G.nodes()])
labels = [G.nodes[i]['club'] for i in G.nodes()]

color_map = {'Mr. Hi': 0, 'Officer': 1}
y = np.array([color_map[label] for label in labels])

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=100)
for i, txt in enumerate(G.nodes()):
    plt.annotate(txt, (X[i, 0], X[i, 1]))
plt.title("Node2Vec Embeddings of Zachary's Karate Club")
plt.show()
