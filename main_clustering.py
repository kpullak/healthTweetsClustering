# -*- coding: utf-8 -*-
"""

Created on Thursday June 18 17:38:40 2020
@author: Krishna

Implementation of the health news tweet clustering using the
Jaccard Distance metric and K-means clustering algorithm to
cluster tweets in the best possible manner and generate the
tweet_cluster mapping csv file as output

"""


import random as rd
import math
import pandas as pd

import matplotlib.pyplot as plt


def k_means(tweets, k = 4, max_iterations=50):

    centroids = []

    # initialization, assign random tweets as centroids
    count = 0
    hash_map = dict()
    while count < k:
        random_tweet_idx = rd.randint(0, len(tweets) - 1)
        if random_tweet_idx not in hash_map:
            count += 1
            hash_map[random_tweet_idx] = True
            centroids.append(tweets[random_tweet_idx])

    iter_count = 0
    prev_centroids = []

    # run the iterations until not converged or until the max iteration in not reached
    while (not is_converged(prev_centroids, centroids)) and (iter_count < max_iterations):

        print("running iteration " + str(iter_count))

        # assignment, assign tweets to the closest centroids
        clusters = assign_cluster(tweets, centroids)

        # to check if k-means converges, keep track of prev_centroids
        prev_centroids = centroids

        # update, update centroid based on clusters formed
        centroids = update_centroids(clusters)
        iter_count = iter_count + 1

    if iter_count == max_iterations:
        print("max iterations reached, K means not converged")
    else:
        print("converged")

    sse = compute_sse(clusters)

    return clusters, sse


def is_converged(prev_centroid, new_centroids):

    # false if lengths are not equal
    if len(prev_centroid) != len(new_centroids):
        return False

    # iterate over each entry of clusters and check if they are same
    for c in range(len(new_centroids)):
        if " ".join(new_centroids[c]) != " ".join(prev_centroid[c]):
            return False

    return True


def assign_cluster(tweets, centroids):

    clusters = dict()

    # for every tweet iterate each centroid and assign closest centroid to a it
    for t in range(len(tweets)):
        min_dis = math.inf
        cluster_idx = -1;
        for c in range(len(centroids)):
            dis = get_distance(centroids[c], tweets[t])
            # look for a closest centroid for a tweet

            if centroids[c] == tweets[t]:
                # print("tweet and centroid are equal with c: " + str(c) + ", t" + str(t))
                cluster_idx = c
                min_dis = 0
                break

            if dis < min_dis:
                cluster_idx = c
                min_dis = dis

        # randomise the centroid assignment to a tweet if nothing is common
        if min_dis == 1:
            cluster_idx = rd.randint(0, len(centroids) - 1)

        # assign the closest centroid to a tweet
        clusters.setdefault(cluster_idx, []).append([tweets[t]])
        # print("tweet t: " + str(t) + " is assigned to cluster c: " + str(cluster_idx))
        # add the tweet distance from its closest centroid to compute sse in the end

        last_tweet_idx = len(clusters.setdefault(cluster_idx, [])) - 1
        clusters.setdefault(cluster_idx, [])[last_tweet_idx].append(min_dis)

    return clusters


def update_centroids(clusters):

    centroids = []

    # iterate each cluster and check for a tweet with closest distance sum with all other tweets in the same cluster
    # select that tweet as the centroid for the cluster
    for c in range(len(clusters)):
        min_dis_sum = math.inf
        centroid_idx = -1

        # to avoid redundant calculations
        min_dis_dp = []

        for t1 in range(len(clusters[c])):
            min_dis_dp.append([])
            dis_sum = 0
            # get distances sum for every of tweet t1 with every tweet t2 in a same cluster
            for t2 in range(len(clusters[c])):
                if t1 != t2:
                    if t2 < t1:
                        dis = min_dis_dp[t2][t1]
                    else:
                        dis = get_distance(clusters[c][t1][0], clusters[c][t2][0])

                    min_dis_dp[t1].append(dis)
                    dis_sum += dis
                else:
                    min_dis_dp[t1].append(0)

            # select the tweet with the minimum distances sum as the centroid for the cluster
            if dis_sum < min_dis_sum:
                min_dis_sum = dis_sum
                centroid_idx = t1

        # append the selected tweet to the centroid list
        centroids.append(clusters[c][centroid_idx][0])

    return centroids


# computing the jaccard distance
def get_distance(tweet1, tweet2):

    # get the intersection
    intersection = set(tweet1).intersection(tweet2)

    # get the union
    union = set().union(tweet1, tweet2)

    # return the jaccard distance
    return 1 - (len(intersection) / len(union))


# computing the sum of squared error terms
def compute_sse(clusters):

    sse = 0
    # iterate every cluster 'c', compute SSE as the sum of square of distances of the tweet from it's centroid
    for c in range(len(clusters)):
        for t in range(len(clusters[c])):
            sse = sse + (clusters[c][t][1] * clusters[c][t][1])

    return sse


if __name__ == '__main__':

    # tweets = open('bbc_processed_tweets.txt', "r", encoding="ISO-8859-1")
    tweets = open('combined_processed_tweets.txt', "r", encoding="ISO-8859-1")
    tweets = list(tweets)

    experiments = 1  # default number of experiments to be performed
    k = 7  # default value of K for K-means
    SSE = []  # list to hold the SSE for different cluster combinations

    # for every experiment 'e', run K-means
    for e in range(experiments):

        print("------ Running K means for experiment no. " + str((e + 1)) + " for k = " + str(k))
        clusters, sse = k_means(tweets, k)
        cluster_new = {}

        # write the mappings to a new dictionary
        for key in clusters:
            cluster_new[key] = []

        # write the cluster mappings to a new dictionary removing the distance to the mean (used for SSE computation)
        for key in clusters:
            values = clusters[key]
            for value in values:
                value_new = value[0]
                cluster_new[key].append(value_new)

        start = [[]]
        for key in cluster_new:
            values = cluster_new[key]
            for value in values:
                # value_str = ' '.join(word for word in value)
                start.append([value, key])

        df = pd.DataFrame(start, columns=['tweet', 'cluster_number'])
        df = df.iloc[1:]  # for the empty row that was created
        df.to_csv("tweet_cluster_mapping.csv", encoding='utf-8', index=False)

        # for every cluster 'c', print size of each cluster
        for c in range(len(clusters)):
            print(str(c+1) + ": ", str(len(clusters[c])) + " tweets")
            # to print tweets in a cluster
            # for t in range(len(clusters[c])):
            #     print("t" + str(t) + ", " + (" ".join(clusters[c][t][0])))

        print("--> SSE : " + str(sse))
        SSE.append(sse)
        print('\n')

        # increment k after every experiment
        k = k + 1

    print('The sum of squared errors for differing number of clusters is:')
    print(SSE)

cluster_number = []
for alpha in range(experiments):
    cluster_number.append(7 + alpha)

fig = plt.figure()
plt.scatter(cluster_number, SSE)
plt.xlabel('Clusters')
plt.ylabel('Sum of Squared Error')
fig.savefig('foo.png')

# based on the SSE values computed above, finding the best cluster_number based on elbow plots
from kneed import KneeLocator
kn = KneeLocator(cluster_number, SSE, curve='convex', direction='decreasing')
print(kn.knee)
# result: the best cluster count is found to be = 7

fig = plt.figure()
plt.xlabel('number of clusters k')
plt.ylabel('Sum of squared distances')
plt.plot(cluster_number, SSE, 'bx-')
plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
# plt.show()
fig.savefig('plt.png')