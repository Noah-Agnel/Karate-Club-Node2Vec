This project uses Zachary's karate club, a graph representing the social network of a karate club that split into two groups after a conflict between the owner and the karate instructor. 
The goal is to use the node2vec approach to embed the nodes into 2-dimensional vectors that will be vizualisable to then assign each member to one of the two groups formed after the conflict using Kmeans clustering.

The overall result obtained is very positive. All members except one are correctly assigned to their group. The member that was incorrectly assigned is member #9. 
After spending some time hyperparameter tuning without avail, I did some research and found out that member #9 was also the only member incorrectly assigned by Wayne W. Zachary in his club assignement.
We can therefore safely assume that the error isn't due to the approach of the hyper-parameter tuning but rather information that isn't represented in the graph structure and edge weights.
