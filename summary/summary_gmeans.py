import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class GMeansClusteringAnalyzer:
    def __init__(
        self,
        base_path="results/g_means_results",
        true_labels=None,
        features=None,
        external_weight=0.5,
    ):
        self.base_path = base_path
        self.true_labels = true_labels
        self.features = features
        self.external_weight = external_weight

    def analyze_dataset(self, dataset_name, external_weight=None):
        if external_weight is not None:
            self.external_weight = external_weight

        files = [
            f"{self.base_path}/{dataset_name}_MinMax_G-Means_metrics.csv",
            f"{self.base_path}/{dataset_name}_Robust_G-Means_metrics.csv",
            f"{self.base_path}/{dataset_name}_Standard_G-Means_metrics.csv",
        ]

        dataframes = []
        for f, scaler in zip(files, ["MinMax", "Robust", "Standard"]):
            df = pd.read_csv(f)
            df["scaler_type"] = scaler
            dataframes.append(df)

        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Combined columns: {combined_df.columns.tolist()}")  # Add this

        try:
            results = self._find_best_clusters(
                combined_df, dataset_name, external_weight=self.external_weight
            )
            if results is None:
                print("_find_best_clusters returned None")
                return None, None
            latex_table = self._generate_latex_table(
                results, dataset_name, external_weight
            )
            return results, latex_table
        except Exception as e:
            print(f"Error in analyze_dataset: {str(e)}")
            return None, None

    def _reorder_confusion_matrix(self, cm):
        n = cm.shape[0]
        cost = -cm  # Convert to cost matrix for minimization
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost)
        return cm[row_ind, :][:, col_ind]

    def _scale_metrics(self, df):
        try:
            scaler = MinMaxScaler()
            intra_metrics = [
                "silhouette_score",
                "davies_bouldin",
                "calinski_harabasz",
            ]
            extra_metrics = [
                "adjusted_rand_index",
                "fowlkes_mallows_index",
            ]

            print(f"Available columns: {df.columns.tolist()}")
            print(f"Looking for metrics: {intra_metrics + extra_metrics}")
            print(
                f"Are all metrics present? {all(metric in df.columns for metric in intra_metrics + extra_metrics)}"
            )

            df_new = df.copy()
            df_new[
                [
                    "silhouette_score_scaled",
                    "davies_bouldin_scaled",
                    "calinski_harabasz_scaled",
                ]
            ] = scaler.fit_transform(df[intra_metrics])
            print("Intra metrics scaled")
            df_new["davies_bouldin_scaled"] = 1 - df_new["davies_bouldin_scaled"]
            df_new[["ari_scaled", "fmi_scaled"]] = scaler.fit_transform(
                df[extra_metrics]
            )
            print("Extra metrics scaled")

            return df_new
        except Exception as e:
            print(f"Error in _scale_metrics: {str(e)}")
            return None

    def _find_best_clusters(
        self, df, dataset_name, additional_params=None, external_weight=0.5
    ):
        df_scaled = None
        if additional_params is None:
            additional_params = [
                "Alpha",
                "distance_metric",
                "Method name",
                "centroids",
            ]

        df_scaled = self._scale_metrics(df)
        if df_scaled is None:  # Add this check
            print("Scaling failed")
            return None

        # Split metrics
        intra_scaled = [
            "silhouette_score_scaled",
            "davies_bouldin_scaled",
            "calinski_harabasz_scaled",
        ]
        extra_scaled = ["ari_scaled", "fmi_scaled"]

        # Calculate weighted scores
        internal_weight = 1 - external_weight
        df_scaled["weighted_score"] = internal_weight * (
            df_scaled[intra_scaled].sum(axis=1) / len(intra_scaled)
        ) + external_weight * (df_scaled[extra_scaled].sum(axis=1) / len(extra_scaled))

        best_per_k = pd.DataFrame()
        for k in sorted(df_scaled["Optimal k"].unique()):
            k_data = df_scaled[df_scaled["Optimal k"] == k]
            best_k = k_data.loc[k_data["weighted_score"].idxmax()]
            best_per_k = pd.concat([best_per_k, pd.DataFrame([best_k])])

        metrics = [
            "silhouette_score",
            "davies_bouldin",
            "calinski_harabasz",
            "adjusted_rand_index",
            "fowlkes_mallows_index",
        ]
        columns = ["Optimal k", "scaler_type"] + additional_params + metrics
        return best_per_k[columns]

    def _generate_latex_table(self, df, dataset_name, external_weight):
        df = df.copy().reset_index(drop=True)

        table = (
            "\\begin{table}[ht]\n"
            "\\centering\n"
            "\\resizebox{\\textwidth}{!}{\n"
            "\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|}\n"
            "\\hline\n"
            "Optimal k & scaler & Alpha & distance & Method & silhouette & davies & calinski & ari & fmi \\\\\n"
            "\\hline\n"
        )

        for _, row in df.iterrows():
            line = [
                str(int(row["Optimal k"])),
                str(row["scaler_type"]),
                f"{row['Alpha']:.4f}",
                str(row["distance_metric"]),
                str(row["Method name"]),
                f"{row['silhouette_score']:.3f}",
                f"{row['davies_bouldin']:.3f}",
                f"{row['calinski_harabasz']:.3f}",
                f"{row['adjusted_rand_index']:.3f}",
                f"{row['fowlkes_mallows_index']:.3f}",
            ]
            table += " & ".join(line) + " \\\\\n\\hline\n"

        table += (
            "\\end{tabular}}\n"
            f"\\caption{{Best {dataset_name} G-Means Results per k (External Weight: {external_weight:.1f})}}\n"
            f"\\label{{tab:{dataset_name}_GMeans}}\n"
            "\\end{table}"
        )

        os.makedirs("summary/Latex_tables", exist_ok=True)
        with open(
            f"summary/Latex_tables/best_{dataset_name}_w{external_weight}_GMeans.txt",
            "w",
        ) as f:
            f.write(table)

        return table

    def _find_and_save_best_clusters(
        self, df, dataset_name, additional_params=None, external_weight=0.5
    ):
        if additional_params is None:
            additional_params = [
                "n_neighbors",
                "eigen_solver",
                "n_init",
                "gamma",
                "execution_time",
            ]

        # Pass additional_params here instead of None
        results = self._find_best_clusters(
            df,
            dataset_name,
            additional_params=additional_params,
            external_weight=external_weight,
        )
        latex_table = self._generate_latex_table(results, dataset_name, external_weight)
        return results, latex_table
