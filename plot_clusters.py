# -*- coding: utf-8 -*-
"""

Created on Thursday June 18 17:38:40 2020
@author: Krishna

This script will read the tweet cluster mapping csv file generated
in the previous step and plots them on a graph
* has capabilities to annotate each data point with the corresponding
* text on the graph, but commented it out for ease of readability

"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# read the tweet_cluster_mapping files using pandas data frame
df = pd.read_csv('tweet_cluster_mapping.csv')
x = df['tweet'].tolist()

# generating the vector representations of the tweets
cv = CountVectorizer(analyzer='word', max_features=5000, lowercase=True,
                     preprocessor=None, tokenizer=None, stop_words='english')
vectors = cv.fit_transform(x)

# performing the k-means cluster analysis with a cluster size of 7 (as determined by our prev analysis)
kmeans = KMeans(n_clusters=7, init='k-means++', random_state=0)
kmean_indices = kmeans.fit_predict(vectors)

# In order to make a plot on a 2D graph, computing the first two principal components
pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(vectors.toarray())

# assigning a different color combination to each of the clusters
colors = ["r", "b", "c", "y", "m", "g", "k"]

x_axis = [o[0] for o in scatter_plot_points]
y_axis = [o[1] for o in scatter_plot_points]
fig, ax = plt.subplots(figsize=(20, 10))

ax.scatter(x_axis, y_axis, c=[colors[d] for d in kmean_indices])

# this is to annotate each data point with the corresponding text -
# (commented this part of the code as it is clogging up the entire graph with annotations)
# for i, txt in enumerate(x):
#     ax.annotate(txt, (x_axis[i], y_axis[i]))

fig.savefig('cluster_points.png')