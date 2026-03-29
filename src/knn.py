import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

from data_loader import load_dataset, prepare_classification_data


def save_metrics(metrics: dict, filepath: str):
    """
    Guarda las métricas del modelo en un archivo de texto.

    Parámetros:
        metrics (dict): Diccionario con nombre y valor de cada métrica.
        filepath (str): Ruta del archivo de salida.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"Métricas guardadas en: {filepath}")


def plot_confusion_matrix(y_test, y_pred, save_path: str):
    """
    Genera y guarda la matriz de confusión como imagen.
    La diagonal principal representa los aciertos del modelo.
    Los valores fuera de la diagonal son errores de clasificación.

    Parámetros:
        y_test: Etiquetas reales.
        y_pred: Etiquetas predichas por el modelo.
        save_path (str): Ruta donde se guarda la imagen.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["GALAXY", "QSO", "STAR"],
        yticklabels=["GALAXY", "QSO", "STAR"],
    )
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión — KNN")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Matriz de confusión guardada en: {save_path}")


def plot_grid_search(
    cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2, save_path: str
):
    """
    Grafica los resultados del GridSearchCV para visualizar
    el efecto de cada combinación de hiperparámetros.

    Parámetros:
        cv_results: Resultados del GridSearchCV (clf.cv_results_).
        grid_param_1: Valores del primer parámetro (n_neighbors).
        grid_param_2: Valores del segundo parámetro (metric).
        name_param_1 (str): Nombre del eje X.
        name_param_2 (str): Nombre de la leyenda.
        save_path (str): Ruta donde se guarda la imagen.
    """
    scores_mean = np.array(cv_results["mean_test_score"]).reshape(
        len(grid_param_2), len(grid_param_1)
    )
    _, ax = plt.subplots(1, 1)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx, :], "-o", label=f"{name_param_2}: {val}")
    ax.set_title("Grid Search Scores — KNN")
    ax.set_xlabel(name_param_1)
    ax.set_ylabel("CV Average Score")
    ax.legend(loc="best")
    ax.grid(True)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Grid search guardado en: {save_path}")


# ──────────────────────────────────────────────
# modelo
# ──────────────────────────────────────────────


def train_knn(X_train, y_train, k: int = 5):
    """
    Entrena un modelo KNN simple con k vecinos fijos.

    Parámetros:
        X_train: Datos de entrenamiento escalados.
        y_train: Etiquetas de entrenamiento.
        k (int): Número de vecinos (default=5 según requerimiento).

    Retorna:
        knn (KNeighborsClassifier): Modelo entrenado.
    """
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn


def evaluate_knn(knn, X_test, y_test):
    """
    Evalúa el modelo KNN sobre el conjunto de prueba.

    Parámetros:
        knn: Modelo KNN entrenado.
        X_test: Datos de prueba escalados.
        y_test: Etiquetas reales del conjunto de prueba.

    Retorna:
        y_pred: Predicciones del modelo.
        acc (float): Accuracy del modelo.
    """
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy KNN (k=5): {acc:.4f}")
    return y_pred, acc


def train_knn_gridsearch(X_train, y_train):
    """
    Entrena un KNN usando GridSearchCV para encontrar
    automáticamente los mejores hiperparámetros (k y métrica de distancia).

    Parámetros:
        X_train: Datos de entrenamiento escalados.
        y_train: Etiquetas de entrenamiento.

    Retorna:
        clf (GridSearchCV): Modelo entrenado con los mejores parámetros.
    """
    parameters = {
        "n_neighbors": (3, 5, 7, 9, 11, 13, 15, 17, 19, 21),
        "metric": ("euclidean", "manhattan"),
    }
    clf = GridSearchCV(
        KNeighborsClassifier(),
        parameters,
        scoring="accuracy",
        cv=5,
        return_train_score=True,
    )
    clf.fit(X_train, y_train)
    print(f"Mejores parámetros: {clf.best_params_}")
    return clf


def save_model(model, filepath: str):
    """
    Guarda el modelo entrenado en disco usando joblib.
    El archivo .pkl permite cargar el modelo después sin reentrenarlo.

    Parámetros:
        model: Modelo sklearn entrenado.
        filepath (str): Ruta de destino del archivo .pkl.
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Modelo guardado en: {filepath}")


# ──────────────────────────────────────────────
# ejecucion principal
# ──────────────────────────────────────────────

if __name__ == "__main__":

    # carga y preparacion de datos
    df = load_dataset("dataset/sdss_sample.csv")
    X_train, X_test, y_train, y_test, scaler = prepare_classification_data(df)

    # modelo KNN
    knn = train_knn(X_train, y_train, k=5)
    y_pred, acc = evaluate_knn(knn, X_test, y_test)

    # guardar metricas
    save_metrics({"Accuracy_KNN_k5": round(acc, 4)}, "outputs/metrics/knn_metrics.txt")

    # guardar matriz de confusion
    plot_confusion_matrix(y_test, y_pred, "outputs/plots/knn_confusion_matrix.png")

    # guardar modelo
    save_model(knn, "models/knn_model.pkl")

    # modelo avanzado con GridSearchCV
    clf = train_knn_gridsearch(X_train, y_train)
    y_pred_grid = clf.predict(X_test)
    acc_grid = accuracy_score(y_test, y_pred_grid)
    print(f"Accuracy KNN GridSearch: {acc_grid:.4f}")

    # guardar metricas GridSearch
    save_metrics(
        {
            "Accuracy_GridSearch": round(acc_grid, 4),
            "Mejor_k": clf.best_params_["n_neighbors"],
            "Mejor_metrica": clf.best_params_["metric"],
        },
        "outputs/metrics/knn_gridsearch_metrics.txt",
    )

    # guardar grafica GridSearch
    plot_grid_search(
        clf.cv_results_,
        [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        ["euclidean", "manhattan"],
        "Número de vecinos",
        "Distancia",
        save_path="outputs/plots/knn_gridsearch.png",
    )

    # guardar modelo GridSearch
    save_model(clf, "models/knn_gridsearch_model.pkl")
