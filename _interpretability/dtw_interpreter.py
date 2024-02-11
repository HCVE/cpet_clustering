import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter
from collections import Counter
from tslearn.metrics import dtw_path


def check_model(m):
    """Check if the model is fitted. If not an assertion error is raised."""

    assert hasattr(m, "medoids_"), "Model is not fitted. A fitted instance should be used to interpret the " \
                                   "clustering results. "


def is_consecutive(points):
    """Function to check if the points are consecutive"""

    if all(np.diff(points) == 1):
        return True
    else:
        return False


def fix_consecutive_error(points, length=5):
    """Isolate the largest sequence of consecutive points to be used in the linear regression model.

    :param points: list with the indexes of the time-stamps with the highest DTW distance
    :param length: int, minimum length of consecutive indexes

    """

    points = np.sort(points)
    corrected_points = []
    for k, g in groupby(enumerate(points), lambda d: d[0] - d[1]):
        temp = list(map(itemgetter(1), g))
        if len(temp) >= length:
            corrected_points.append(temp)
    return corrected_points


def separate_per_variable(res_dict: dict, n_vars: int):
    for key in res_dict.keys():
        tmp = res_dict[key]
        new_list = [[] for _ in range(n_vars)]
        for j in range(n_vars):
            for i in range(j, len(tmp), 5):
                new_list[j].append(tmp[i])
            new_list[j] = fix_consecutive_error(sum(sum(new_list[j], []), []), length=5)
        res_dict[key] = new_list
    return res_dict


def histogram_threshold(points: list, data_shape: tuple, occurrences):
    # find the number of occurrences for each position
    hist = Counter(points)

    # keep those that have more than a user-defined threshold. If the threshold is integer
    # then the parsed value is used directly. If it is float, then it represents the percentage of the
    # recordings not assigned in the cluster under investigation
    if type(occurrences) == int:
        threshold = occurrences
        samples = [k for k, v in hist.items() if v >= threshold]
    elif type(occurrences) == float:
        threshold = int(data_shape[0]*occurrences)
        samples = [k for k, v in hist.items() if v >= threshold]
    else:
        raise TypeError(f"Variable 'n_occurrences' should of type 'int' or 'float'. "
                        f"{type(occurrences)} has passed instead")
    return samples


class DTWInterpreter:
    """Implementation of the interpretability method for the DTW with k-medoids clustering.

         Parameters
        ----------
        X : array-like of shape (n_examples, n_variables, n_samples)
            The time series dataset used for training the clustering model and to be used in the interpretability
            of the model

        clusters : array or list with the labels of each training example

        clustering_model : the clustering model used to retrieve the time series clusters

        threshold : float, optional, default 0.5
            the 1-qth quantile of the maximum per sample distance calculated using the DTW.
            The value should be between 0 and 1

        n_occurrences : int or float, optional, default : 0.5
            If float the minimum number of occurrences of the most distinguishable time-stamps as percentage
            of the total recordings in the dataset. If int the minimum number of occurrences.

        minimum_window_size : int, optional, default 3
            The minimum length of the consecutive indexes with the most separable time-stamps

        cluster_samples: dictionary, optional, default None
            A dictionary containing the new numbering order of the clusters.
            This is used to ensure that cluster 1 is the one with the most favourable profile
        """

    def __init__(self, X, clusters, clustering_model, threshold=0.5, n_occurrences=0.5, minimum_window_size=3,
                 cluster_names=None):
        self.ts_data = X
        self.clusters = clusters
        self.model = clustering_model
        self.quantile_threshold = threshold
        self.n_occurrences = n_occurrences
        self.window_size = minimum_window_size

        if cluster_names is None:
            self.cluster_names = {i: i for i in np.unique(clusters)}
        else:
            self.cluster_names = self._reverse_cluster_labels(cluster_names)

        check_model(self.model)

    @staticmethod
    def _reverse_cluster_labels(cl_names):
        return dict((v, k) for k, v in cl_names.items())

    @staticmethod
    def _get_quantile(dist, threshold=0.5):
        qnt = np.quantile(dist, threshold)
        return np.where(dist >= qnt)[0]

    def sample_wise_distances(self, data, centroid):
        paths = np.empty((data.shape[0]), dtype=np.ndarray)
        sample_wise_distances = np.empty((data.shape[0]), dtype=np.ndarray)

        for n_examples in range(data.shape[0]):
            p, dd = dtw_path(data[n_examples], centroid,
                             global_constraint=self.model.global_constraint,
                             sakoe_chiba_radius=self.model.sakoe_chiba_radius,
                             itakura_max_slope=self.model.itakura_slope)

            paths[n_examples] = p
            dist = [np.linalg.norm(data[n_examples, pos[0], :] - centroid[pos[1], :]) for pos in p]
            sample_wise_distances[n_examples] = dist
        return paths, sample_wise_distances

    def interpret_results(self):
        cluster_assignments = np.unique(self.clusters)
        res = {f"cluster {i}": [] for i in cluster_assignments}

        for cl in tqdm(cluster_assignments, total=len(cluster_assignments)):
            # get the medoid as calculated by the clustering algorithm

            # get the time-series recordings that were not assigned in the cluster under investigation
            pos_outside_cluster = np.array([p for p, i in enumerate(self.clusters) if i != cl])
            other_cluster_data = self.ts_data[pos_outside_cluster, :, :]

            pos_in_cluster = np.array([p for p, i in enumerate(self.clusters) if i == cl])
            in_cluster_data = self.ts_data[pos_in_cluster, :, :]

            # get the warping paths and the sample-wise distances between
            # the data not assigned in the cluster under investigation and the centroid of that cluster.
            for data_index in range(len(in_cluster_data)):
                warping_paths, d = self.sample_wise_distances(data=other_cluster_data,
                                                              centroid=in_cluster_data[data_index])

                # for n_var in range(d.shape[-1]):
                important_points = []
                for n_examples in range(d.shape[0]):
                    # get the positions of the samples that resulted in the highest dtw distance
                    max_distance_pos = self._get_quantile(d[n_examples], threshold=self.quantile_threshold)
                    tmp = np.array(warping_paths[n_examples])
                    max_distance_medoid_pos = [t[1] for t in tmp[max_distance_pos]]
                    important_points.append(max_distance_medoid_pos)

                # flatten the list with the points for all examples
                important_point_flattened = sum(important_points, [])

                # keep those that have more than a user-defined threshold. If the threshold is integer
                # then the parsed value is used directly. If it is float, then it represents the percentage of the
                # recordings not assigned in the cluster under investigation
                samples = histogram_threshold(points=important_point_flattened, data_shape=d.shape,
                                              occurrences=self.n_occurrences)
                res[f"cluster {cl}"].append(fix_consecutive_error(samples, length=self.window_size))

            tmp_list = sum(sum(res[f"cluster {cl}"], []), [])
            samples = histogram_threshold(points=tmp_list, data_shape=in_cluster_data.shape,
                                          occurrences=self.n_occurrences)
            res[f"cluster {cl}"] = fix_consecutive_error(samples, length=self.window_size)
        return res


if __name__ == "__main__":
    import os
    import pickle
    from tqdm import tqdm
    from _utils.load_data import fetch_data
    from tslearn.utils import to_time_series_dataset
    from _utils.utils import visualize_most_important_window

    data_path = os.path.join("C:/Users/vagge/Desktop/PhD/CPET/Data/iCOMPEER")
    sex_file = os.path.join("C:/Users/vagge/Desktop/PhD/CPET/Data/s64901_06JUN2023_all.xlsx")
    save_data_path = os.path.join("C:/Users/vagge/Desktop/PhD/CPET/Results")
    male_cpet, female_cpet = fetch_data(data_path, sex_file)

    variables = ["HR", "V'O2", "RER", "PETO2", "PETCO2"]
    sex = "females"
    cpet = {"males": male_cpet, "females": female_cpet}

    ts_data = []
    patient_ids = []
    for ind in tqdm(range(len(cpet[sex]))):
        ts_data.append(cpet[sex]["CPET Data"].iloc[ind][variables].to_numpy())
        patient_ids.append(cpet[sex]["Patient IDs"].iloc[ind])
    formatted_dataset = to_time_series_dataset(ts_data)

    model = pickle.load(open(r"C:\Users\vagge\Desktop\PhD\CPET\Results\dtw_kmedoids\females\5 clusters\model.pkl", "rb"))
    results = pd.read_excel(
        r"C:\Users\vagge\Desktop\PhD\CPET\Results\dtw_kmedoids\females\5 clusters\clustering_assignments_females.xlsx")

    cluster_labels = {0: 3, 1: 5, 2: 1, 3: 4, 4: 2}
    cluster_colours = {1: "green", 2: "blue", 3: "darkorange", 4: "blueviolet", 5: "red"}

    wnds = DTWInterpreter(X=formatted_dataset, clusters=results["Cluster"], clustering_model=model,
                          threshold=0.5, n_occurrences=0.5, minimum_window_size=5,
                          cluster_names=cluster_labels).interpret_results()
    visualize_most_important_window(cpet_data=cpet[sex], model=model, ids=patient_ids, res=wnds,
                                    cl_labels=cluster_labels, cl_colours=cluster_colours)
    print("Finished")
