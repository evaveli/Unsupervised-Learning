import os
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import anderson
from .k_means import KMeans
from metrics.evaluation_metrics import MetricCollection
from utils import save_metrics_to_csv, print_heading2

class Cluster:
    """
    Represents a cluster containing data points and their indices.
    """
    def __init__(self, data, idxs):
        self.data = data
        self.idxs = idxs
        self.centroid = None


class GMeans:
    """
    Implementation of the G-Means clustering algorithm.
    """

    def __init__(
        self,
        max_k=None,
        random_state=None,
        tol=1e-4,
        distance_metric="euclidean",
        alpha=0.05,
        plus_plus=False,
    ):
        """
        Initialize the GMeans object.
        """
        self.max_k = max_k
        self.random_state = random_state
        self.tol = tol
        self.distance_metric = distance_metric
        self.alpha = alpha
        self.plus_plus = plus_plus
        self.cluster_centroids = None
        self.labels = None

    def _gaussian_test(self, data):
        """
        Perform a Gaussianity test on the given data using PCA and the Anderson-Darling test.
        """
        if data.shape[0] < 2:
            # Not enough data to perform PCA and Anderson-Darling test -> Gaussian
            return True

        # Check for zero variance in case of identical data in a cluster
        if np.all(np.var(data, axis=0) == 0):
            # Zero variance along all dimensions, cannot perform PCA
            return True  # Consider data Gaussian to prevent further splitting

        # Project the data along the main principal component
        pca = PCA(n_components=1)
        dim_reduced_data = pca.fit_transform(data).flatten()

        # Anderson Test
        result = anderson(dim_reduced_data, dist="norm")

        # Obtain significance level and critical values
        significance_levels = result.significance_level
        critical_values = result.critical_values

        # Get the statistically critical values
        idxs = np.abs(significance_levels - (self.alpha * 100)).argmin()
        critical_value = critical_values[idxs]

        # Compare test statistic with critical value
        if result.statistic < critical_value:
            return True  # Gaussian
        else:
            return False  # Not Gaussian

    def _process_cluster(self, cluster):
        """
        Process a cluster to check if it should be split further.
        """
        data = cluster.data
        idxs = cluster.idxs

        # If the cluster has less than 2 samples, do not split further
        if data.shape[0] < 2:
            cluster.centroid = data[0]
            return {"cluster": cluster, "split": False}

        perform_kmeans = KMeans()

        # Obtain the center of all data
        centroid, _, _ = perform_kmeans.k_means(
            X=data,
            K=1,
            distance_metric=self.distance_metric,
            plus_plus=self.plus_plus,
            random_state=self.random_state,
        )

        # Set as center of all data
        cluster.centroid = centroid[0]

        # Check if data is Gaussian distributed
        is_gaussian = self._gaussian_test(data=data)

        if is_gaussian:
            # Return as it is
            return {"cluster": cluster, "split": False}
        else:
            # Cluster not Gaussian distributed, so split in two
            _, labels, _ = perform_kmeans.k_means(
                X=data,
                K=2,
                distance_metric=self.distance_metric,
                plus_plus=self.plus_plus,
                random_state=self.random_state,
            )

            # Create two new cluster objects
            new_clusters = []
            for i in range(2):
                idxs_mask = labels == i
                new_data = data[idxs_mask]
                new_idxs = idxs[idxs_mask]

                if new_data.shape[0] > 0:
                    new_cluster = Cluster(data=new_data, idxs=new_idxs)
                    new_clusters.append(new_cluster)
            if len(new_clusters) == 2:
                # Append also cluster in case we reached max_k
                return {"clusters": new_clusters, "split": True, "cluster": cluster}
            else:
                # Cannot split further
                return {"cluster": cluster, "split": False}

    def gmeans(self, data):
        """
        Perform G-Means clustering on the given dataset.
        """

        n_samples, _ = data.shape

        # Create a cluster object containing all samples
        initial_cluster = Cluster(data=data, idxs=np.arange(n_samples))
        clusters_to_check = [initial_cluster]
        final_clusters = []
        number_clusters = 1

        while clusters_to_check:
            new_clusters = []
            results = []

            # Obtain clusters
            for cluster in clusters_to_check:
                result = self._process_cluster(cluster)
                results.append(result)

            # Create final cluster list
            for res in results:
                if not res["split"]:
                    final_clusters.append(
                        res["cluster"]
                    )  # If Gaussian, directly append cluster
                else:
                    # Two free spots until max_k is reached
                    if self.max_k is not None and number_clusters + 1 > self.max_k:
                        final_clusters.append(res["cluster"])
                    else:
                        new_clusters.extend(res["clusters"])
                        number_clusters += 1

            # Check the two new clusters if still enough spots until max_k
            clusters_to_check = new_clusters

            # Fill the two last spots in the final_clusters list
            if self.max_k is not None and number_clusters > self.max_k:
                final_clusters.extend(clusters_to_check)
                break

        # Collect cluster centers and labels
        self.cluster_centroids = np.array(
            [cluster.idxs[0] for cluster in final_clusters]
        )
        # Assign labels to all data points
        self.labels = np.empty(n_samples, dtype=int)
        for i, cluster in enumerate(final_clusters):
            self.labels[cluster.idxs] = i

        return self.cluster_centroids, self.labels, number_clusters


def create_results_dict(
    results,
    metric,
    initial_centroid_idxs,
    metric_collection,
    scaling,
    clustering_algo,
    final_k,
    alpha,
):
    results[metric][alpha] = {
        "Dataset scaling": scaling,
        "Method name": clustering_algo,
        "Optimal k": final_k,
        "centroids": (
            initial_centroid_idxs.tolist()
            if isinstance(initial_centroid_idxs, np.ndarray)
            else initial_centroid_idxs
        ),
        "silhouette_score": float(metric_collection.silhouette),
        "davies_bouldin": float(metric_collection.davies_bouldin),
        "calinski_harabasz": float(metric_collection.calinski_harabasz),
        "adjusted_rand_index": float(metric_collection.ari),
        "fowlkes_mallows_index": float(metric_collection.fmi),
        "normalized_mutual_info": float(metric_collection.nmi),
        "purity_score": float(metric_collection.purity),
    }
    return results


def run_gmeans_experiment(data, y_true, dataset_choice, clustering_algo, dataset_name):

    if dataset_choice == 1:
        scaling = "MinMax"
    elif dataset_choice == 2:
        scaling = "Robust"
    elif dataset_choice == 3:
        scaling = "Standard"

    distance_metrics = ["euclidean", "manhattan", "cosine"]

    # Hardcoded the highest k out of the predefined lists
    if dataset_name == "cmc":
        max_k = 8
    elif dataset_name == "hepatitis":
        max_k = 8
    elif dataset_name == "pen-based":
        max_k = 12

    significance_levels = [0.0001, 0.0005, 0.005, 0.05]

    # Iterate over three distance measures
    results = {}

    for d_metric in distance_metrics:
        results[d_metric] = {}
        for alpha in significance_levels:
            print_heading2(f"Running G-Means Analysis for metric: {d_metric}")

            gmeans_algorithm = GMeans(
                max_k=max_k, distance_metric=d_metric, alpha=alpha
            )
            cluster_centroids, labels, final_k = gmeans_algorithm.gmeans(data)
            print(f"\nG-Means found optimal K = {final_k} clusters")

            y_true_filtered = np.array(y_true, dtype=int)
            y_pred_filtered = np.array(labels, dtype=int)

            metric_collection = MetricCollection(
                X=data,
                best_labels=labels,
                metric=d_metric,
                y_true_filtered=y_true_filtered,
                y_pred_filtered=y_pred_filtered,
            )

            metric_collection.map_labels()  # Map labels for confusion matrix and purity

            metric_collection.calculate_internal_indexes()
            metric_collection.calculate_external_indexes()
            metric_collection.print_evaluation()

            results = create_results_dict(
                results=results,
                metric=d_metric,
                initial_centroid_idxs=cluster_centroids,
                metric_collection=metric_collection,
                scaling=scaling,
                clustering_algo=clustering_algo,
                final_k=final_k,
                alpha=alpha,
            )

            print(results)

    save_metrics_to_csv(
        output_dir=os.path.join(os.getcwd(), "results", "g_means_results"),
        results=results,
        dataset_name=f"{dataset_name}_{scaling}_{clustering_algo}",
        k_alorithm_flag=False,
    )
