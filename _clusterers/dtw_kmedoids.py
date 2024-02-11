import os
import pickle
from copy import deepcopy

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids
from tslearn.metrics import cdist_dtw, dtw
from _clusterers.dunn_index import dunn_fast


class DTWCluster:
    """Implementation of the k-medoids clustering algorithm combined with the Dynamic time warping (DTW).

     Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of medoids to
        generate.

    metric : string, or callable, optional, default: 'precomputed'
        What distance metric to use. See :func:metrics.pairwise_distances
        metric can be 'precomputed', the user must then feed the fit method
        with a precomputed kernel matrix and not the design matrix X.

    method : {'alternate', 'pam'}, default: 'pam'
        Which algorithm to use. 'alternate' is faster while 'pam' is more accurate.

    init : {'random', 'heuristic', 'k-medoids++', 'build'}, or array-like of shape
        (n_clusters, n_features), optional, default: 'k-medoids++'
        Specify medoid initialization method. 'random' selects n_clusters
        elements from the dataset. 'heuristic' picks the n_clusters points
        with the smallest sum distance to every other point. 'k-medoids++'
        follows an approach based on k-means++_, and in general, gives initial
        medoids which are more separated than those generated by the other methods.
        'build' is a greedy initialization of the medoids used in the original PAM
        algorithm. Often 'build' is more efficient but slower than other
        initializations on big datasets and it is also very non-robust,
        if there are outliers in the dataset, use another initialization.
        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        .. _k-means++: https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf

    max_iter : int, optional, default : 30
        Specify the maximum number of iterations when fitting. It can be zero in
        which case only the initialization is computed which may be suitable for
        large datasets when the initialization is sufficiently efficient
        (i.e. for 'build' init).

    random_state : int, RandomState instance or None, optional
        Specify random state for the random number generator. Used to
        initialise medoids when init='random'.

    Attributes
    ----------
    medoids_ : array, shape = (n_clusters, n_variables, n_samples)
        Cluster centers, i.e. medoids (elements from the original dataset)

    labels_ : array, shape = (n_samples,)
        Labels of each point
    """
    def __init__(self, n_clusters, normalize=False, global_constraint=None, itakura_max_slope=None,
                 sakoe_chiba_radius=None, metric="precomputed", method="pam", max_iter=30, init="k-medoids++",
                 random_state=0):
        self.labels_ = None
        self.medoids_ = None
        self.normalize = normalize
        self.global_constraint = global_constraint
        self.itakura_slope = itakura_max_slope
        self.sakoe_chiba_radius = sakoe_chiba_radius
        self.distance_matrix = None
        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state
        self.clustering_model = KMedoids(n_clusters=self.n_clusters, metric=self.metric, method=self.method,
                                         max_iter=self.max_iter, init=self.init, random_state=self.random_state)

    @staticmethod
    def _normalize_time_series(dataset):
        normalized_dataset = deepcopy(dataset)

        for var in range(normalized_dataset.shape[-1]):
            scaler = MinMaxScaler().fit(normalized_dataset[:, :, var].flatten().reshape((-1, 1)))
            for n_example in range(normalized_dataset.shape[0]):
                normalized_dataset[n_example, :, var] = np.squeeze(
                    scaler.transform(normalized_dataset[n_example, :, var].reshape(-1, 1)))
        return normalized_dataset

    def dtw_distance_matrix(self, dataset):
        distance = cdist_dtw(dataset, global_constraint=self.global_constraint, itakura_max_slope=self.itakura_slope,
                             sakoe_chiba_radius=self.sakoe_chiba_radius)

        # correct in case of infinite values --> make them 10 times larger than the max value
        max_value = np.max(distance[~np.isinf(distance)])
        distance[np.isinf(distance)] = 10 * max_value
        return distance

    def _get_distance_from_medoid(self, training_data):
        for n_example in range(training_data.shape[0]):
            distance = []
            for n_medoid in range(self.medoids_.shape[0]):
                distance.append(dtw(s1=training_data[n_example], s2=self.medoids_[n_medoid]))
            yield np.argmin(distance)

    def fit(self, X):
        if self.normalize:
            X = self._normalize_time_series(dataset=X)
        distance_matrix = self.dtw_distance_matrix(dataset=X)
        self.clustering_model.fit(distance_matrix)
        medoids_indices = self.clustering_model.medoid_indices_
        self.medoids_ = X[medoids_indices, :, :]
        self.labels_ = self.clustering_model.labels_
        return self

    def fit_predict(self, X):
        if self.normalize:
            X = self._normalize_time_series(dataset=X)
        distance_matrix = self.dtw_distance_matrix(dataset=X)
        self.clustering_model.fit_predict(distance_matrix)
        medoids_indices = self.clustering_model.medoid_indices_
        self.medoids_ = X[medoids_indices, :, :]
        self.labels_ = self.clustering_model.labels_
        return self.labels_

    def predict(self, X):
        assert self.medoids_ is not None
        return list(self._get_distance_from_medoid(training_data=X))

    @staticmethod
    def check_path(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def save(self, path):
        self.check_path(path)
        pickle.dump(self, open(os.path.join(path, "model.pkl"), "wb"))

    def return_cvi(self, X):
        if self.normalize:
            X = self._normalize_time_series(dataset=X)
        distance_matrix = self.dtw_distance_matrix(X)

        self.clustering_model.fit(distance_matrix)
        inertia = self.clustering_model.inertia_
        silhouette = silhouette_score(distance_matrix, metric="precomputed", labels=self.clustering_model.labels_)
        dunn = dunn_fast(distance_matrix, self.clustering_model.labels_)
        return {"inertia": inertia, "silhouette": silhouette, "dunn": dunn}, self.clustering_model.labels_
