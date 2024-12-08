import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
)
from sklearn.metrics import confusion_matrix
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=UserWarning)


class SpectralClusteringAnalysis:
    def __init__(self, dataset_name):

        self.affinity_types = ["nearest_neighbors", "rbf"]
        self.eigen_solver_types = ["arpack", "lobpcg"]

        # Dataset-specific parameters with extended ranges
        if dataset_name == "hepatitis":
            self.n_clusters_range = [2, 3, 4, 5, 6, 7, 8]
            self.n_neighbors_range = [5]
            self.gamma_range = [0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55]
            self.n_init_range = [3, 5, 7]
            self.assign_labels_types = [
                "kmeans",
                "cluster_qr",
            ]  # WE keep both to verify
            self.eigen_solver_types = ["arpack"]
            self.affinity_types = ["rbf"]
            self.kernels = ["nearest_neighbors"]

        elif dataset_name == "pen-based":
            self.n_clusters_range = [8, 9, 10, 11, 12]
            self.n_neighbors_range = [25]
            self.gamma_range = [0.005, 0.0075, 0.01, 0.0125, 0.015]
            self.assign_labels_types = ["kmeans", "cluster_qr"]
            self.eigen_solver_types = [
                "arpack"
            ]  # This was the best in preliminary searches
            self.n_init_range = [40, 80, 120]
            self.kernels = ["nearest_neighbors"]

        elif dataset_name == "cmc":
            self.n_clusters_range = [2, 3, 4, 5, 6, 7, 8]
            self.n_neighbors_range = [25]
            self.gamma_range = [0.01, 0.4, 1.0]
            self.assign_labels_types = ["kmeans", "cluster_qr"]
            self.eigen_solver_types = [
                "arpack"
            ]  # This was the best in preliminary searches
            self.n_init_range = [40]
            self.kernels = ["nearest_neighbors"]

    def perform_spectral_clustering(
        self,
        X,
        y_true,
        binary_vars,
        categorical_vars,
        dataset_choice,
        dataset_name=None,
    ):
        """
        Perform Spectral Clustering with different parameters and evaluate results
        """
        print("Starting Spectral Clustering Analysis...")
        results = []

        if dataset_choice == 1:
            scaling = "MinMax"
        elif dataset_choice == 2:
            scaling = "Robust"
        elif dataset_choice == 3:
            scaling = "Standard"

        if len(binary_vars) > 0 and len(categorical_vars) > 0:
            # If we have binary variables, create both datasets
            binary_indices = list(range(len(binary_vars)))
            cat_indices = list(range(len(categorical_vars)))
            remove_indices = binary_indices + cat_indices
            nonbin_X = np.delete(X, binary_indices, axis=1)
            noncat_X = np.delete(X, cat_indices, axis=1)
            cont_X = np.delete(X, remove_indices, axis=1)
            pca = PCA(n_components=0.95)  # Keep 95% of variance
            X_pca_nonbin = pca.fit_transform(nonbin_X)
            X_pca_cont = pca.fit_transform(cont_X)
            datasets = {
                "full": X,
                "PCA_non_binary": X_pca_nonbin,
                "PCA_cont": X_pca_cont,
                "no_binary": nonbin_X,
                "no_cat": noncat_X,
                "Just continuous": cont_X,
            }
        elif len(binary_vars) > 0 and len(categorical_vars) == 0:
            # If we have binary variables, create both datasets
            binary_indices = list(range(len(binary_vars)))
            remove_indices = binary_indices
            cont_X = np.delete(X, remove_indices, axis=1)
            pca = PCA(n_components=0.95)
            X_pca_cont = pca.fit_transform(cont_X)
            datasets = {"full": X, "PCA_cont": X_pca_cont, "Just continuous": cont_X}
        elif len(binary_vars) == 0 and len(categorical_vars) > 0:
            # If we have binary variables, create both datasets
            cat_indices = list(range(len(categorical_vars)))
            remove_indices = cat_indices
            cont_X = np.delete(X, remove_indices, axis=1)
            pca = PCA(n_components=0.95)  # Keep 95% of variance
            X_pca_cont = pca.fit_transform(cont_X)
            datasets = {"full": X, "PCA_cont": X_pca_cont, "Just continuous": cont_X}
        else:
            # If no binary variables, just use the full dataset
            pca = PCA(n_components=0.95)  # Keep 95% of variance
            X_pca = pca.fit_transform(X)
            pca90 = PCA(n_components=0.90)  # Keep 95% of variance
            X_pca90 = pca90.fit_transform(X)
            umap2 = UMAP(n_components=2, random_state=42)
            X_umap2 = umap2.fit_transform(X)
            datasets = {"full": X, "PCA": X_pca, "PCA90": X_pca90, "UMAP2": X_umap2}

            # For nearest_neighbors affinity
        for dataset_type, data in datasets.items():
            for n_clusters in self.n_clusters_range:
                for n_neighbors in self.n_neighbors_range:
                    for eigen_solver in self.eigen_solver_types:
                        for assign_labels in self.assign_labels_types:
                            for n_init in self.n_init_range:
                                for kernel in self.kernels:
                                    start_time = time.time()
                                    try:
                                        spectral = SpectralClustering(
                                            n_clusters=n_clusters,
                                            affinity=kernel,
                                            n_neighbors=n_neighbors,
                                            eigen_solver=eigen_solver,
                                            assign_labels=assign_labels,
                                            n_init=n_init,
                                            random_state=42,
                                            n_jobs=-1,
                                        )

                                        labels = spectral.fit_predict(X)
                                        execution_time = time.time() - start_time

                                        scores = self._calculate_scores(
                                            X, labels, y_true
                                        )
                                        scores.update(
                                            {
                                                "dataset_type": dataset_type,
                                                "n_clusters": n_clusters,
                                                "affinity": "nearest_neighbors",
                                                "n_neighbors": n_neighbors,
                                                "eigen_solver": eigen_solver,
                                                "assign_labels": assign_labels,
                                                "n_init": n_init,
                                                "gamma": "N/A",
                                                "execution_time": execution_time,
                                            }
                                        )

                                        results.append(scores)

                                    except Exception as e:
                                        self._print_error(
                                            "nearest_neighbors",
                                            n_clusters,
                                            n_neighbors,
                                            eigen_solver,
                                            e,
                                        )

        # For rbf affinity
        for n_clusters in self.n_clusters_range:
            for gamma in self.gamma_range:
                for eigen_solver in self.eigen_solver_types:
                    for assign_labels in self.assign_labels_types:
                        for n_init in self.n_init_range:
                            start_time = time.time()
                            try:
                                spectral = SpectralClustering(
                                    n_clusters=n_clusters,
                                    affinity="rbf",
                                    gamma=gamma,
                                    eigen_solver=eigen_solver,
                                    assign_labels=assign_labels,
                                    n_init=n_init,
                                    random_state=42,
                                    n_jobs=-1,
                                )

                                labels = spectral.fit_predict(X)
                                execution_time = time.time() - start_time

                                scores = self._calculate_scores(X, labels, y_true)

                                scores.update(
                                    {
                                        "dataset_type": dataset_type,
                                        "n_clusters": n_clusters,
                                        "affinity": "rbf",
                                        "n_neighbors": "N/A",
                                        "eigen_solver": eigen_solver,
                                        "assign_labels": assign_labels,
                                        "n_init": n_init,
                                        "gamma": str(gamma),
                                        "execution_time": execution_time,
                                    }
                                )

                                results.append(scores)

                            except Exception as e:
                                self._print_error(
                                    "rbf", n_clusters, gamma, eigen_solver, e
                                )

        results_df = pd.DataFrame(results)

        # Make syre this exists
        os.makedirs("results/spectral", exist_ok=True)

        print("Spectral finished, see the results in the result folder")

        results_df.to_csv(
            f"results/spectral/{dataset_name}_{scaling}_spectral_clustering_results.csv",
            index=False,
        )

        # If UMAP was used, return both
        if "UMAP2" in datasets:
            return results_df, X_umap2
        else:
            return results_df, None

    def _calculate_scores(self, X, labels, y_true):

        # Calculate internal metrics
        internal_scores = {
            "silhouette_score": silhouette_score(X, labels),
            "davies_bouldin_score": davies_bouldin_score(X, labels),
            "calinski_harabasz_score": calinski_harabasz_score(X, labels),
        }

        # Calculate external metrics
        external_scores = {
            "adjusted_rand_index": adjusted_rand_score(y_true, labels),
            "normalized_mutual_info": normalized_mutual_info_score(y_true, labels),
            "fowlkes_mallows_index": fowlkes_mallows_score(y_true, labels),
        }

        # Calculate purity
        contingency_matrix = pd.crosstab(y_true, labels).values
        purity_score = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(
            contingency_matrix
        )

        # Combine all scores
        scores = {**internal_scores, **external_scores}
        scores["purity_score"] = purity_score

        return scores

    def _print_results(self, scores):
        print(f"\nCompleted clustering with parameters:")
        for key, value in scores.items():
            print(f"{key}: {value}")

    def _print_error(self, affinity_type, n_clusters, param, eigen_solver, error):
        print(f"\nError with parameters:")
        print(f"affinity: {affinity_type}")
        print(f"n_clusters: {n_clusters}")
        print(f"param (n_neighbors/gamma): {param}")
        print(f"eigen_solver: {eigen_solver}")
        print(f"Error message: {str(error)}")

    def plot_umap_comparison(
        self, X_umap, predicted_labels, true_labels, title="UMAP Projections Comparison"
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot predicted clusters
        scatter1 = ax1.scatter(
            X_umap[:, 0], X_umap[:, 1], c=predicted_labels, cmap="tab10"
        )
        ax1.set_title("UMAP with Predicted Clusters")
        ax1.set_xlabel("UMAP1")
        ax1.set_ylabel("UMAP2")
        plt.colorbar(scatter1, ax=ax1)

        # Plot true labels
        scatter2 = ax2.scatter(X_umap[:, 0], X_umap[:, 1], c=true_labels, cmap="tab10")
        ax2.set_title("UMAP with True Labels")
        ax2.set_xlabel("UMAP1")
        ax2.set_ylabel("UMAP2")
        plt.colorbar(scatter2, ax=ax2)

        plt.suptitle(title)
        plt.savefig("umap_comparison.png")
        plt.close()
