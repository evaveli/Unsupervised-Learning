import os
import time
import pandas as pd
import sys
from preprocessing import preprocessing
from cluster_algorithms import optics
from cluster_algorithms.k_means import KMeans
from cluster_algorithms.fuzzy_c_means import CMeans
from cluster_algorithms.g_means import run_gmeans_experiment
from cluster_algorithms.spectral import SpectralClusteringAnalysis
from summary.summary_spectral import SpectralClusteringAnalyzer
from summary.summary_optics import OPTICSAnalyzer
from summary.summary_gmeans import GMeansClusteringAnalyzer
from summary.summary_cmeans import CMeansAnalyzer
from plotting.spaghetti_plot import CreatesSpaghettiPlot
from utils import load_datasets


def display_menu():
    print("\nClustering Algorithm Selection:")
    print("1) OPTICS algorithm")
    print("2) Spectral algorithm")
    print("3) K-means")
    print("4) K-means++")
    print("5) G-Means")
    print("6) Fuzzy C-means")
    print(
        "7) Generate LateX Code/ Confusion matrix for Optics | Spectral | G-Means | C-Means"
    )
    print("8) Create Spaggetti Plots for C-Means | K-Means | K-Means++ | G-Means")
    print("9) Exit")
    return input("\nEnter your choice (1-9): ")


def display_menu_dataset_choice():
    print("Please select one of these options:")
    print("1) I would like to use the MinMax Scaled dataset")
    print("2) I would like to use the Robust Scaled dataset")
    print("3) I would like to use the Standard Scaled dataset")
    return input("\nEnter your choice (1-3): ")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "datasets")

    try:
        dataset_name = (
            input("Enter the name of the dataset ['cmc', 'hepatitis', 'pen-based']: ")
            or "cmc"
        )
        if dataset_name not in ["cmc", "hepatitis", "pen-based"]:
            raise ValueError(f"Dataset must be one of: 'cmc', 'hepatitis', 'pen-based'")

    except ValueError as e:
        print(f"You might have misspelled the dataset name: {e}")
        exit(1)

    start_time = time.time()

    preprocessor = preprocessing.Preprocessing()

    # Load dataset
    df = pd.DataFrame(load_datasets(dataset_dir, dataset_name))

    # Preprocess the data
    X, X_robust, X_standard, y_true = preprocessor.generous_preprocessing(df)

    binary_vars = preprocessor.binary_vars
    categorical_vars = preprocessor.categorical_vars
    while True:
        choice = display_menu()

        choice = int(choice)
        if choice == 9:
            print("Exiting the program...")
            sys.exit()

        elif choice == 1:  # optics
            optics_clustering = optics.Optics()
            dataset_choice = display_menu_dataset_choice()
            dataset_choice = int(dataset_choice)
            if dataset_choice == 1:
                optics_clustering.perform_optics(
                    dataset_name, dataset_choice, X, y_true
                )

            elif dataset_choice == 2:

                optics_clustering.perform_optics(
                    dataset_name, dataset_choice, X, y_true
                )
            elif dataset_choice == 3:

                optics_clustering.perform_optics(
                    dataset_name, dataset_choice, X, y_true
                )

        elif choice == 2:  # Spectral
            dataset_choice = (
                display_menu_dataset_choice()
            )  # Call the function to get the choice

            dataset_choice = int(dataset_choice)

            if dataset_choice == 1:
                spectral_analysis = SpectralClusteringAnalysis(dataset_name)
                results = spectral_analysis.perform_spectral_clustering(
                    X,
                    y_true,
                    preprocessor.binary_vars,
                    preprocessor.categorical_vars,
                    dataset_choice,
                    dataset_name=dataset_name,
                )

            elif dataset_choice == 2:
                spectral_analysis = SpectralClusteringAnalysis(dataset_name)
                results = spectral_analysis.perform_spectral_clustering(
                    X_robust,
                    y_true,
                    preprocessor.binary_vars,
                    preprocessor.categorical_vars,
                    dataset_choice,
                    dataset_name=dataset_name,
                )

            elif dataset_choice == 3:
                spectral_analysis = SpectralClusteringAnalysis(dataset_name)
                results = spectral_analysis.perform_spectral_clustering(
                    X_standard,
                    y_true,
                    preprocessor.binary_vars,
                    preprocessor.categorical_vars,
                    dataset_choice,
                    dataset_name=dataset_name,
                )

        elif choice == 3:
            # K-means implementation
            perform_kmeans = KMeans()
            dataset_choice = display_menu_dataset_choice()  # get the choice
            dataset_choice = int(dataset_choice)
            clustering_algo = "K-Means"
            if dataset_choice == 1:
                results = perform_kmeans.run_k_means_experiments(
                    X,
                    dataset_choice,
                    dataset_name=dataset_name,
                    clustering_algo=clustering_algo,
                    y_true=y_true,
                )

            elif dataset_choice == 2:
                results = perform_kmeans.run_k_means_experiments(
                    X_robust,
                    dataset_choice,
                    dataset_name=dataset_name,
                    clustering_algo=clustering_algo,
                    y_true=y_true,
                )
            elif dataset_choice == 3:
                results = perform_kmeans.run_k_means_experiments(
                    X_standard,
                    dataset_choice,
                    dataset_name=dataset_name,
                    clustering_algo=clustering_algo,
                    y_true=y_true,
                )

        # End of Simple K - means
        #

        elif choice == 4:
            # K ++
            perform_kmeans = KMeans()
            dataset_choice = display_menu_dataset_choice()  # get the choice
            dataset_choice = int(dataset_choice)
            clustering_algo = "K-Means++"
            if dataset_choice == 1:
                results = perform_kmeans.run_k_means_experiments(
                    X,
                    dataset_choice,
                    dataset_name=dataset_name,
                    clustering_algo=clustering_algo,
                    y_true=y_true,
                    plus_plus=True,
                )

            elif dataset_choice == 2:
                results = perform_kmeans.run_k_means_experiments(
                    X_robust,
                    dataset_choice,
                    dataset_name=dataset_name,
                    clustering_algo=clustering_algo,
                    y_true=y_true,
                    plus_plus=True,
                )
            elif dataset_choice == 3:
                results = perform_kmeans.run_k_means_experiments(
                    X_standard,
                    dataset_choice,
                    dataset_name=dataset_name,
                    clustering_algo=clustering_algo,
                    y_true=y_true,
                    plus_plus=True,
                )

        # End of K ++

        elif choice == 5:
            # G-Means
            clustering_algo = "G-Means"
            dataset_choice = (
                display_menu_dataset_choice()
            )  # Call the function to get the choice
            dataset_choice = int(dataset_choice)

            if dataset_choice == 1:
                run_gmeans_experiment(
                    data=X,
                    y_true=y_true,
                    dataset_choice=dataset_choice,
                    dataset_name=dataset_name,
                    clustering_algo=clustering_algo,
                )

            elif dataset_choice == 2:
                run_gmeans_experiment(
                    data=X,
                    y_true=y_true,
                    dataset_choice=dataset_choice,
                    dataset_name=dataset_name,
                    clustering_algo=clustering_algo,
                )

            elif dataset_choice == 3:
                run_gmeans_experiment(
                    data=X,
                    y_true=y_true,
                    dataset_choice=dataset_choice,
                    dataset_name=dataset_name,
                    clustering_algo=clustering_algo,
                )
        # End of G-means

        elif choice == 6:  # C++
            # C-means implementation
            perform_cmeans = CMeans()
            dataset_choice = display_menu_dataset_choice()  # get the choice
            dataset_choice = int(dataset_choice)
            clustering_algo = "C-Means"
            if dataset_choice == 1:
                results = perform_cmeans.run_c_means_experiments(
                    X,
                    dataset_choice,
                    dataset_name=dataset_name,
                    clustering_algo=clustering_algo,
                    y_true=y_true,
                )

            elif dataset_choice == 2:
                results = perform_cmeans.run_c_means_experiments(
                    X_robust,
                    dataset_choice,
                    dataset_name=dataset_name,
                    clustering_algo=clustering_algo,
                    y_true=y_true,
                )
            elif dataset_choice == 3:
                results = perform_cmeans.run_c_means_experiments(
                    X_standard,
                    dataset_choice,
                    dataset_name=dataset_name,
                    clustering_algo=clustering_algo,
                    y_true=y_true,
                )

            # End of C-means

        elif choice == 7:
            algorithm = input(
                "Choose clustering algorithm (1: OPTICS, 2: Spectral, 3: G-Means, 4: C-Means): "
            )
            if algorithm not in ["1", "2", "3", "4"]:
                print("Invalid algorithm choice")
                continue

            while True:
                try:
                    weight = float(input("Enter external weight (0-1): "))
                    if 0 <= weight <= 1:
                        break
                    print("Weight must be between 0 and 1")
                except ValueError:
                    print("Please enter a valid number")

            if algorithm == "1":
                analyzer = OPTICSAnalyzer(true_labels=y_true, features=X)
                print("Creating LaTeX code and confusion matrix...")
                results, latex = analyzer.analyze_dataset(
                    dataset_name, external_weight=weight
                )
                print("Finished")
            elif algorithm == "2":
                analyzer = SpectralClusteringAnalyzer(true_labels=y_true, features=X)
                print("Creating LaTeX code and confusion matrix...")
                results, latex = analyzer.analyze_dataset(
                    dataset_name, external_weight=weight
                )
                print("Finished")
            elif algorithm == "3":
                analyzer = GMeansClusteringAnalyzer(true_labels=y_true, features=X)
                print("Creating LaTeX code...")
                results, latex = analyzer.analyze_dataset(
                    dataset_name, external_weight=weight
                )
                print("Finished")

            elif algorithm == "4":
                analyzer = CMeansAnalyzer(true_labels=y_true, features=X)
                print("Creating LaTeX code...")
                results, latex = analyzer.analyze_dataset(
                    dataset_name, external_weight=weight
                )
                print("Finished")

        elif choice == 8:

            algorithm = input(
                "Choose clustering algorithm (1: K-Means, 2: K-Means++, 3: G-Means, 4: C-means): "
            )
            if algorithm not in ["1", "2", "3", "4"]:
                print("Invalid algorithm choice")
                continue

            # Map numeric choice to algorithm name
            algo_map = {
                "1": "K-Means",
                "2": "K-Means++",
                "3": "G-Means",
                "4": "C-Means",
            }
            selected_algorithm = algo_map[algorithm]

            print(f"Creating spaghetti plots for {selected_algorithm}...")
            plotter = CreatesSpaghettiPlot(algorithm=selected_algorithm)
            plotter.create_visualization(dataset_name)
            print("Finished creating spaghetti plots")

        print(f"Time taken: {time.time() - start_time:.2f} seconds")
