import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
from scipy.io import loadmat

def initialize_centroids(X, k):
    """X: dataset, k: number of centroids"""

    number_of_samples = X.shape[0]
    sample_points_ids = random.sample(range(0, number_of_samples), k)

    centroids = [tuple(X[id]) for id in sample_points_ids]
    unique_centroids = list(set(centroids))

    number_of_unique_centroids = len(unique_centroids)

    while number_of_unique_centroids < k:
        new_sample_points_ids = random.sample(range(0, number_of_samples), k - number_of_unique_centroids)
        new_centroids = [tuple(X[id]) for id in new_sample_points_ids]
        unique_centroids = list(set(unique_centroids + new_centroids))

        number_of_unique_centroids = len(unique_centroids)

    return np.array(unique_centroids)

def euclidean_distance(x,y):
    """Computes euclidean distance between matrices x and y."""

    x_sq = np.reshape(np.sum(x * x, axis=1), (x.shape[0], 1))
    y_sq = np.reshape(np.sum(y * y, axis=1), (y.shape[0]), 1)
    xy = x @ y.T

    dist = x_sq + y_sq - 2 * xy

    return np.sqrt(dist)

def assign_clusters(X, centroids):
    """ assigns the points to one of the centroids"""

    k = centroids.shape[0]

    clusters = {}

    distance_matrix = euclidean_distance(X, centroids)

    closest_cluster_ids = np.argmin(distance_matrix, axis=1)

    for i in range(k):
        clusters[i] = []

    for i, cluster_id in enumerate(closest_cluster_ids):
        clusters[cluster_id].append(X[i])

    return clusters

def stop_iteration(old_centroids, new_centroids, threshold):
    ''' stopping condition'''
    dist = euclidean_distance(old_centroids, new_centroids)
    centroids_stop= np.max(dist.diagonal()) <= threshold

    return centroids_stop

def kmeans(X, k,threshold=0):
    """Performs k-means and find k cluster centroid"""

    new_centroids = initialize_centroids(X=X, k=k)

    centroids_covered = False

    while not centroids_covered:
        previous_centroids = new_centroids
        clusters = assign_clusters(X, previous_centroids)

        new_centroids = np.array([np.mean(clusters[key], axis=0, dtype=X.dtype) for key in sorted(clusters.keys())])

        centroids_covered = stop_iteration(previous_centroids, new_centroids, threshold)

    return new_centroids

k = 8
x1 = loadmat('clustering_data2.mat')
data = x1['X']
X = data.T

centroids = kmeans(X, k, threshold=1/100)

clusters = assign_clusters(X, centroids)

marker = itertools.cycle(('+','o','^','s','*','x','p','H')) 

plt.figure()
for centroid, points in clusters.items():
    points = np.array(points)
    centroid = np.mean(points, axis=0)

    plt.scatter(points[:, 0], points[:, 1], marker=next(marker), s = 50)
    plt.grid()
    plt.scatter(centroid[0], centroid[1], marker='x', color="black")

plt.show()