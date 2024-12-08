import os
import numpy as np
from metrics.distance_metrics import (
    euclidean_distance,
    manhattan_distance,
    cosine_distance,
)
from metrics.evaluation_metrics import MetricCollection
from itertools import product
from utils import save_metrics_to_csv, print_heading2


class CMeans:

    def initialize_membership_matrix(self, n_samples, C, random_state=None):
        """Initialize membership matrix with random values that sum to 1 for each sample"""
        if random_state is not None:
            np.random.seed(random_state)

        # Generate random values and normalize columns to sum to 1
        membership_matrix = np.random.random((n_samples, C))
        return membership_matrix / membership_matrix.sum(axis=1)[:, np.newaxis]

    def compute_distances(self, X, centroids, distance_metric):
        """Compute distances between all points and centroids using vectorized operations"""
        if distance_metric == "euclidean":
            distances = np.array(
                [euclidean_distance(X, centroid) for centroid in centroids]
            ).T
        elif distance_metric == "manhattan":
            distances = np.array(
                [manhattan_distance(X, centroid) for centroid in centroids]
            ).T
        elif distance_metric == "cosine":
            distances = np.array(
                [cosine_distance(X, centroid) for centroid in centroids]
            ).T
        else:
            raise ValueError("Unsupported distance metric.")
        return distances

    def update_membership_matrix(self, distances, m):
        """Update membership matrix using vectorized operations"""
        power = -2 / (m - 1)
        distances_power = distances**power
        return distances_power / distances_power.sum(axis=1)[:, np.newaxis]

    def update_centroids(self, X, membership_matrix, m):
        """Update centroids using vectorized operations"""
        weights = membership_matrix**m
        return (weights.T @ X) / weights.sum(axis=0)[:, np.newaxis]

    def compute_objective_function(self, distances, membership_matrix, m):
        """Compute the objective function value"""
        return np.sum((membership_matrix**m) * distances)

    def c_means(
        self,
        X,
        C,
        m=2.0,
        distance_metric="euclidean",
        max_iters=100,
        tol=1e-6,
        random_state=None,
    ):
        """
        Fuzzy C-Means clustering algorithm.
        """
        n_samples, n_features = X.shape

        # Initialize membership matrix
        membership_matrix = self.initialize_membership_matrix(
            n_samples, C, random_state
        )

        prev_obj = float("inf")

        for iteration in range(max_iters):
            # Update centroids
            centroids = self.update_centroids(X, membership_matrix, m)

            # Compute distances
            distances = self.compute_distances(X, centroids, distance_metric)

            # Update membership matrix
            membership_matrix = self.update_membership_matrix(distances, m)

            # Compute objective function
            obj = self.compute_objective_function(distances, membership_matrix, m)

            # Check convergence
            if abs(prev_obj - obj) < tol:
                print(f"Converged in {iteration+1} iterations.")
                break

            prev_obj = obj

        return centroids, membership_matrix

    def create_results_dict(
        self,
        results,
        metric,
        C,
        obj_value,
        metric_collection,
        scaling_method,
        clustering_algo,
        X,
        best_centroids,
        best_membership,
        m,
    ):
        results[metric][C] = {
            "Dataset scaling": scaling_method,
            "Method name": clustering_algo,
            "objective_function": float(obj_value),
            "silhouette_score": float(metric_collection.silhouette),
            "davies_bouldin": float(metric_collection.davies_bouldin),
            "calinski_harabasz": float(metric_collection.calinski_harabasz),
            "xie_beni_index": float(
                self.compute_xie_beni_index(X, best_centroids, best_membership, m)
            ),
            "partition_coefficient": float(
                self.compute_partition_coefficient(best_membership)
            ),
            "partition_entropy": float(self.compute_partition_entropy(best_membership)),
        }
        return results

    def compute_xie_beni_index(self, X, centroids, membership_matrix, m):
        """
        Compute the Xie-Beni index for fuzzy clustering validation.

        XB = Sum(membership_matrix[i,j]^m * ||x_i - c_j||^2) / (n * min||c_i - c_j||^2)
        where n is number of samples, c_i and c_j are different centroids

        Lower values indicate better clustering.

        """
        n_samples = X.shape[0]

        # Compute numerator: total fuzzy variation within clusters
        numerator = 0
        for i in range(n_samples):
            for j in range(len(centroids)):
                numerator += (membership_matrix[i, j] ** m) * np.sum(
                    (X[i] - centroids[j]) ** 2
                )

        # Compute minimum distance between centroids
        min_centroid_dist = float("inf")
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.sum((centroids[i] - centroids[j]) ** 2)
                min_centroid_dist = min(min_centroid_dist, dist)

        # Avoid division by zero
        if min_centroid_dist == 0:
            return float("inf")

        # Calculate final index
        xb_index = numerator / (n_samples * min_centroid_dist)

        return xb_index

    def compute_partition_coefficient(self, membership_matrix):
        """
        Calculate the Partition Coefficient (PC) index.
        PC ranges from 1/c to 1, where c is the number of clusters.
        Higher values indicate better clustering.
        """
        return np.mean(membership_matrix**2)

    def compute_partition_entropy(self, membership_matrix, epsilon=1e-10):
        """
        Calculate the Partition Entropy (PE) index.
        PE ranges from 0 to log(c), where c is the number of clusters.
        Lower values indicate better clustering.
        """
        safe_membership = np.clip(membership_matrix, epsilon, 1.0)
        return -np.mean(membership_matrix * np.log(safe_membership))

    def print_all_metrics(self, result):
        """Print all metrics in a nicely formatted way"""
        print("\n============================================================")
        print("Configuration:")
        print(f"Dataset Scaling:        {result['Dataset scaling']}")
        print(f"Method:                 {result['Method name']}")
        print(f"Distance Metric:        {result['distance_metric']}")
        print(f"Number of Clusters:     {result['num_clusters']}")
        print(f"Fuzziness Parameter:    {result['fuzziness']}")

        print("\nInternal Validation Indices:")
        print(f"Silhouette Score:       {result['silhouette_score']:.4f}")
        print(f"Davies-Bouldin:         {result['davies_bouldin']:.4f}")
        print(f"Calinski-Harabasz:      {result['calinski_harabasz']:.4f}")

        print("\nFuzzy Validation Indices:")
        print(f"Objective Function:      {result['objective_function']:.4f}")
        print(f"Xie-Beni Index:         {result['xie_beni_index']:.4f}")
        print(f"Partition Coefficient:   {result['partition_coefficient']:.4f}")
        print(f"Partition Entropy:       {result['partition_entropy']:.4f}")
        print("============================================================\n")

    def run_c_means_experiments(
        self,
        X,
        dataset_choice,
        clustering_algo,
        y_true,
        dataset_name=None,
        random_state=None,
    ):
        if dataset_choice == 1:
            scaling_method = "MinMax"
        elif dataset_choice == 2:
            scaling_method = "Robust"
        elif dataset_choice == 3:
            scaling_method = "Standard"

        # Define parameter grids for each dataset
        if dataset_name == "cmc":
            parameter_grid = {
                "C_values": [2, 3, 4, 5, 6, 7, 8],
                "m_values": [1.5, 2.0, 2.5, 3.0],
                "max_iters": 100,
                "gt_n_classes": 3,
                "n_initialization": 15,
            }
        elif dataset_name == "hepatitis":
            parameter_grid = {
                "C_values": [2, 3, 4, 5, 6, 7, 8],
                "m_values": [1.3, 1.7, 2.0, 2.3],
                "max_iters": 100,
                "gt_n_classes": 2,
                "n_initialization": 25,
            }
        elif dataset_name == "pen-based":
            parameter_grid = {
                "C_values": [8, 9, 10, 11, 12],
                "m_values": [1.8, 2.2, 2.6, 3.0],
                "max_iters": 200,
                "gt_n_classes": 10,
                "n_initialization": 10,
            }
        elif dataset_name is None:
            print("Please provide a valid dataset name")
            return "Please provide a valid dataset name"

        distance_metrics = ["euclidean", "manhattan", "cosine"]
        # Initialize results dictionary properly
        all_results = []

        # Create grid of all parameter combinations
        param_combinations = list(
            product(
                ["euclidean", "manhattan", "cosine"],
                parameter_grid["C_values"],
                parameter_grid["m_values"],
            )
        )

        # Iterate through all parameter combinations
        for metric, C, m in param_combinations:
            print_heading2(
                f"Running C-Means Analysis for C: {C}, metric: {metric}, m: {m}"
            )

            best_obj = float("inf")
            best_membership = None
            best_centroids = None

            # Multiple initializations for each parameter combination
            base_seed = 42
            for i in range(parameter_grid["n_initialization"]):
                print(f"\nIteration number: {i+1}")
                iteration_seed = (
                    None if random_state is None else (base_seed + i * 1000)
                )

                centroids, membership_matrix = self.c_means(
                    X,
                    C=C,
                    m=m,
                    distance_metric=metric,
                    max_iters=parameter_grid["max_iters"],
                    random_state=iteration_seed,
                )

                # Compute objective function
                distances = self.compute_distances(X, centroids, metric)
                obj = self.compute_objective_function(distances, membership_matrix, m)
                print(f"Objective function value: {obj}, m: {m}")

                if obj < best_obj:
                    best_obj = obj
                    best_membership = membership_matrix
                    best_centroids = centroids

            # Convert fuzzy membership to crisp labels for evaluation
            best_labels = np.argmax(best_membership, axis=1)

            y_true_filtered = np.array(y_true, dtype=int)
            y_pred_filtered = np.array(best_labels, dtype=int)

            metric_collection = MetricCollection(
                X=X,
                best_labels=best_labels,
                metric=metric,
                y_true_filtered=y_true_filtered,
                y_pred_filtered=y_pred_filtered,
            )

            metric_collection.map_labels()  # Map labels for confusion matrix and purity

            # Compute confusion matrix only when number of clusters corresponds to the true value
            if C == parameter_grid["gt_n_classes"]:
                metric_collection.compute_confusion(
                    C, metric, dataset_name, scaling_method, clustering_algo
                )

            metric_collection.calculate_internal_indexes()

            #  append each result to our list
            result = {
                "Dataset scaling": scaling_method,
                "Method name": clustering_algo,
                "distance_metric": metric,
                "num_clusters": C,
                "fuzziness": m,
                "objective_function": float(best_obj),
                "silhouette_score": float(metric_collection.silhouette),
                "davies_bouldin": float(metric_collection.davies_bouldin),
                "calinski_harabasz": float(metric_collection.calinski_harabasz),
                "xie_beni_index": float(
                    self.compute_xie_beni_index(X, best_centroids, best_membership, m)
                ),
                "partition_coefficient": float(
                    self.compute_partition_coefficient(best_membership)
                ),
                "partition_entropy": float(
                    self.compute_partition_entropy(best_membership)
                ),
            }
            self.print_all_metrics(result)
            all_results.append(result)

        print(f"Total number of experiments: {len(all_results)}")

        # NOW create the final structure for saving
        final_results = {"euclidean": {}, "manhattan": {}, "cosine": {}}
        metric_counters = {"euclidean": 0, "manhattan": 0, "cosine": 0}

        for result in all_results:
            metric = result["distance_metric"]
            idx = str(metric_counters[metric])
            final_results[metric][idx] = result
            metric_counters[metric] += 1

        print(f"Results for euclidean: {len(final_results['euclidean'])}")
        print(f"Results for manhattan: {len(final_results['manhattan'])}")
        print(f"Results for cosine: {len(final_results['cosine'])}")

        # Save the properly structured results
        save_metrics_to_csv(
            output_dir=os.path.join(os.getcwd(), "results", "c_means_results"),
            results=final_results,
            dataset_name=f"{dataset_name}_{scaling_method}_{clustering_algo}",
            k_alorithm_flag=False,
        )

        return final_results
