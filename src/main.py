"""
main.py
-------
Script principal del proyecto ML-ASTRONOMIC.
Ejecuta en orden todos los módulos del pipeline:
  1. Análisis exploratorio
  2. Clasificación KNN
  3. Regresión Lineal
  4. Clustering KMeans

Este es el punto de entrada para Docker y Jenkins.
"""

import sys
import os

# Agregar carpeta src al path para encontrar los módulos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

# Crear carpetas de salida si no existen
Path("outputs/plots").mkdir(parents=True, exist_ok=True)
Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)


def run_data_analysis():
    print("\n" + "=" * 50)
    print("  ANÁLISIS EXPLORATORIO")
    print("=" * 50)
    from data_loader import (
        load_dataset,
        print_basic_info,
        print_statistics,
        print_correlations,
        plot_distributions,
    )

    df = load_dataset("dataset/sdss_sample.csv")
    print_basic_info(df)
    print_statistics(df)
    print_correlations(df)
    plot_distributions(df, "outputs/plots/data_distribution.png")


def run_knn():
    print("\n" + "=" * 50)
    print("  CLASIFICACIÓN — KNN")
    print("=" * 50)
    from data_loader import load_dataset, prepare_classification_data
    from knn import (
        train_knn,
        evaluate_knn,
        plot_confusion_matrix,
        save_metrics,
        save_model,
        train_knn_gridsearch,
        plot_grid_search,
    )
    from sklearn.metrics import accuracy_score

    df = load_dataset("dataset/sdss_sample.csv")
    X_train, X_test, y_train, y_test, scaler = prepare_classification_data(df)

    # Modelo simple k=5
    knn = train_knn(X_train, y_train, k=5)
    y_pred, acc = evaluate_knn(knn, X_test, y_test)
    save_metrics({"Accuracy_KNN_k5": round(acc, 4)}, "outputs/metrics/knn_metrics.txt")
    plot_confusion_matrix(y_test, y_pred, "outputs/plots/knn_confusion_matrix.png")
    save_model(knn, "models/knn_model.pkl")

    # Modelo avanzado GridSearch
    clf = train_knn_gridsearch(X_train, y_train)
    y_pred_grid = clf.predict(X_test)
    acc_grid = accuracy_score(y_test, y_pred_grid)
    save_metrics(
        {
            "Accuracy_GridSearch": round(acc_grid, 4),
            "Mejor_k": clf.best_params_["n_neighbors"],
            "Mejor_metrica": clf.best_params_["metric"],
        },
        "outputs/metrics/knn_gridsearch_metrics.txt",
    )
    plot_grid_search(
        clf.cv_results_,
        [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        ["euclidean", "manhattan"],
        "Número de vecinos",
        "Distancia",
        save_path="outputs/plots/knn_gridsearch.png",
    )
    save_model(clf, "models/knn_gridsearch_model.pkl")


def run_linear_regression():
    print("\n" + "=" * 50)
    print("  REGRESIÓN LINEAL")
    print("=" * 50)
    from data_loader import load_dataset, prepare_regression_data
    from linear_regression import (
        train_linear_regression,
        evaluate_regression,
        plot_real_vs_predicted,
        save_metrics,
        save_model,
    )

    df = load_dataset("dataset/sdss_sample.csv")
    X_train, X_test, y_train, y_test, scaler = prepare_regression_data(df)

    model = train_linear_regression(X_train, y_train)
    y_pred, mse, r2 = evaluate_regression(model, X_test, y_test)
    save_metrics(
        {"MSE": round(mse, 4), "R2": round(r2, 4)},
        "outputs/metrics/linear_regression_metrics.txt",
    )
    plot_real_vs_predicted(y_test, y_pred, "outputs/plots/linear_regression.png")
    save_model(model, "models/linear_regression_model.pkl")


def run_kmeans():
    print("\n" + "=" * 50)
    print("  CLUSTERING — KMEANS")
    print("=" * 50)
    from data_loader import load_dataset, prepare_clustering_data
    from kmeans import train_kmeans, predict_clusters, plot_clusters_vs_real, save_model

    df = load_dataset("dataset/sdss_sample.csv")
    X_train, X_test, y_test, scaler = prepare_clustering_data(df)

    kmeans = train_kmeans(X_train, n_clusters=3)
    clusters = predict_clusters(kmeans, X_test)
    plot_clusters_vs_real(X_test, clusters, y_test, "outputs/plots/kmeans_clusters.png")
    save_model(kmeans, "models/kmeans_model.pkl")


if __name__ == "__main__":
    print("\n🚀 Iniciando pipeline ML-ASTRONOMIC...\n")

    run_data_analysis()
    run_knn()
    run_linear_regression()
    run_kmeans()

    print("\n✅ Pipeline completado exitosamente.")
    print("   Métricas  → outputs/metrics/")
    print("   Gráficas  → outputs/plots/")
    print("   Modelos   → models/")
