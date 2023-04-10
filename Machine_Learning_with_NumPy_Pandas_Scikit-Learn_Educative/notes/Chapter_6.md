# Clustering with scikit-learn
1. [Introduction](#introduction)
2. [Cosine Similarity](#cosine-similarity)
3. [Nearest Neighbors](#nearest-neighbors)
4. [K-Means Clustering](#k-means-clustering)
5. [Hierarchical Clustering](#hierarchical-clustering)
6. [Mean Shift Clustering](#mean-shift-clustering)
7. [DBSCAN](#dbscan)
8. [Evaluating Clusters](#evaluating-clusters)
9. [Feature Clustering](#feature-clustering)
10. Quiz

## Introduction
- Unsupervised learning methods - methods of extracting insights from unlabeled datasets.

- ### A. Unsupervised learning
    - Unsupervised learning methods are centered around finding similarities/differences between data observations and making inferences based on those findings.
    - [Clustering](https://en.wikipedia.org/wiki/Cluster_analysis):
        - Most commonly used form of unsupervised learning.

## Cosine Similarity
- ### Chapter Goals
    - Understand how the cosine similarity metric measures the similarity between two data observations.

- ### A. What defines similarity?
    - [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity):
        - The most common measurement of similarity.
        - Range: -1 to 1
        - Values:
            - Closer to 1: Greater similarity between the observations.
            - Closer to -1: Represent divergence.
            - 0: Two data observations have no correlation (neither similar nor dissimilar).

- ### B. Calculating cosine similarity
    - Calculated as the [dot product](https://en.wikipedia.org/wiki/Euclidean_vector#Dot_product) between the L2-normalization of the vectors.
    - Scikit-learn's [metrics.pairwise.cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) function.
    - Calculates cosine similarities for:
        - Pairs of data observations in a single dataset.
        - Pairs of data observations between two datasets.

- ### Time to Code!
    - [numpy.fill_diagonal](https://numpy.org/doc/stable/reference/generated/numpy.fill_diagonal.html)

## Nearest Neighbors
- ### Chapter Goals
    - Learn how to find the nearest neighbors for a data observation.

- ### A. Finding the nearest neighbors
    - Scikit-learn's [neighbors.NearestNeighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors)

## K-Means Clustering
- ### Chapter Goals
    - Lean about K-means clustering and how it works.
    - Understand why mini-batch clustering is used for large datasets.

- ### A. K-Means algorithm
    - [K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering): Most well-known clustering method.
    - *Centroids*: Cluster means
        - Represent the "centers" of each cluster.
        - A cluster's centroid is equal to the average of all the data observations within the cluster.
    - Initialization of cluster centroids:
        - Randomly initialized
        - (Better) initialized using the [K-means++](https://en.wikipedia.org/wiki/K-means%2B%2B) algorithm
    - Scikit-learn's [cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

- ### B. Mini-batch clustering
    - When working with very large datasets, to reduce the computation time, K-means clustering is applied to randomly sampled subsets of the data (mini-batches) at a time.
    - Scikit-learn's [cluster.MiniBatchKMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html#sklearn.cluster.MiniBatchKMeans)

- ### Time to Code!

## Hierarchical Clustering
- ### Chapter Goals
    - Learn about hierarchical clustering using the agglomerative approach.

- ### A. K-means vs hierarchical clustering
    - Assumption in K-means clustering:
        - A major assumption is that the dataset consists of spherical (i.e. circular) clusters.
            - As real life data often does not contain spherical clusters, K-means clustering might end up producing inaccurate clusters due to its assumption.
    - [Hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering):
        - An alternative to K-means clustering.
        - Allows us to cluster any type of data
            - It doesn't make any assumptions about the data or clusters.
        - Two approaches:
            - Bottom-up (divisive)
                - Initially treats all the data as single cluter.
                - Then repeatedly splits it into smaller clusters until we reach the desired number of clusters.
            - Top-down (agglomerative)
                - Initially treats each data observation as its own cluster.
                - Then repeatedly merges the two most similar clusters until we reach the desired number of clusters.

- ### B. Agglomerative clustering
    - Scikit-learn's [cluster.AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering)
        - [Usage guide](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
            - Describes the linkage criteria that determines the metric used for the merge strategy.
    - Doesn't make use of centroids.
        - No ```predict``` function for making cluster predictions on new data.

## Mean Shift Clustering
- Use mean shift clustering to determine the optimal number of clusters.

- ### Chapter Goals
    - Learn about the [mean shift](https://en.wikipedia.org/wiki/Mean_shift) clustering algorithm.

- ### A. Choosing the number of clusters
    - Like the K-means clustering algorithm, the mean shift algorithm is based on finding cluster centroids.
    - The algorithm looks for "blobs" in the data that can be potential candidates for clusters.
    - Using these "blobs", the algorithm finds a number of candidate centroids.
    - It then removes the candidates that are basically duplicates of others to form the final set of centroids.
    - Scikit-learn's [cluster.MeanShift](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html)
        - [User guide](https://scikit-learn.org/stable/modules/clustering.html#mean-shift)

## DBSCAN
- ### Chapter Goals
    - Learn about the DBSCAN algorithm.

- ### A. Clustering by density
    - Drawbacks of mean shift clustering
        - Not very scalable due to computation time.
        - Assumption: Clusters have "blob" like shape.
            - This assumption is not as strong as the one made by K-means.
    - [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) autoomatically chooses the number of clusters.
    - DBSCAN clusters data by finding dense regions in the dataset.
        - *High-density regions*: Regions in the dataset with many closely packed data observations.
            - The algorithm treats high-density regions as clusters in the dataset.
            - These regions are defined by *core samples*, which are h=just data observations with many neighbors.
        - *Low-density regions*: Regions with sparse data.
            - Observations in the low-density regions are treated as noise and not placed in a cluster.
    - Highly scalable.
    - Makes no assumptions about the underlying shape of clusters in the dataset.

- ### B. Neighbors and core samples
    - Specify:
        - ${\epsilon}$: Maximum distance between two data observations that are considered neighbors.
        - Minimum number of points in the neighborhood of a data observation for the observation to be considered a core sample.
    - Scikit-learn's [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN)
        - DBSCAN: Density-Based Spatial Clustering of Applications with Noise.
        - [User guide](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
            - Clusters found by DBSCAN can be any shape, as opposed by K-Means which assumes that clusters are convex shaped.

## Evaluating Clusters
- ### Chapter Goals
    - Learn how to evaluate clustering algorithms.

- ### A. Evaluation metrics
    - Access to any true cluster assignments (labels)?
        - **No**: Visualize and see if they make sense with respect to the dataset and domain.
        - **Yes**: Apply a number of metrics to evaluate our clustering algorithm.
    - [Adjusted Rand index](https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index)(ARI)
        - [Rand index](https://en.wikipedia.org/wiki/Rand_index#Rand_index)
            - $R = (a+b)/(a+b+c+d)$
            - Intuitively,
                - $a+b$ Number of agreements between X and Y.
                - $c+d$ Number of disagreements between X and Y.
        - ARI value range: [-1, 1]
            - Negative score: Bad labeling
            - Score: 1 :: Perfect labeling
            - Random labeling get a score near 0.
        - Scikit-learn's [metrics.adjusted_rand_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html#sklearn.metrics.adjusted_rand_score)
    - [Adjusted Mutual Information](https://en.wikipedia.org/wiki/Adjusted_mutual_information)(AMI)
        - Scikit-learn's [cluster.adjusted_mutual_info_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score)
        - Metrics similar to ARI.
    - When to use which?
        - ARI:
            - True clusters are large and approximately equal sized.
        - AMI:
            - True cluseters are unbalanced in size and there exist small clusters.

## Feature Clustering
- ### Chapter Goals
    - Learn how to use agglomerative clustering for feature dimensionality reduction.

- ### A. Agglomerative feature clustering
    - In [Data Preprocessing](./Chapter_4.md#pca), PCA was used to perform feature dimensionality reduction.
    - Feature dimensionality reduction can also be performed using agglomerative clustering.
    - Features are reduced by merging common features into clusters.
    - Scikit-learn's [cluster.FeatureAgglomeration](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html#sklearn.cluster.FeatureAgglomeration)
