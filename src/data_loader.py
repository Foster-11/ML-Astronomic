"""
data_loader.py
--------------
Módulo central del proyecto ML-ASTRONOMIC.
Encargado de cargar, preprocesar y analizar el dataset SDSS (Sloan Digital Sky Survey).
Proporciona funciones reutilizables para todos los modelos del proyecto.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ──────────────────────────────────────────────
# Carga del dataset
# ──────────────────────────────────────────────


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Carga el dataset SDSS desde un archivo CSV.

    Parámetros:
        filepath (str): Ruta al archivo CSV.

    Retorna:
        pd.DataFrame: Dataset cargado.
    """
    df = pd.read_csv(filepath)
    return df


# ──────────────────────────────────────────────
# Análisis exploratorio
# ──────────────────────────────────────────────


def print_basic_info(df: pd.DataFrame):
    """
    Imprime información básica del dataset:
    número de filas, columnas, tipos de datos y valores nulos.
    """
    print("=" * 50)
    print(f"Forma del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
    print("=" * 50)
    print("\nInformación general:")
    df.info()
    print("\nPrimeras 5 filas:")
    print(df.head())


def print_statistics(df: pd.DataFrame):
    """
    Imprime estadísticas descriptivas del dataset:
    media, desviación estándar, mínimo, máximo y percentiles.
    """
    print("\nEstadísticas descriptivas:")
    print(df.describe())


def print_correlations(df: pd.DataFrame):
    """
    Imprime las correlaciones de todas las variables numéricas
    con la variable objetivo redshift, ordenadas de mayor a menor.
    """
    print("\nCorrelaciones con redshift:")
    print(df.corr(numeric_only=True)["redshift"].sort_values(ascending=False))


def plot_distributions(df: pd.DataFrame, save_path: str):
    """
    Genera y guarda los histogramas de distribución de todas las variables.
    """
    df.hist(bins=20, figsize=(12, 8))
    plt.suptitle("Distribución de Variables — SDSS", y=1.02)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Distribuciones guardadas en: {save_path}")


# ──────────────────────────────────────────────
# Preparación de datos para cada modelo
# ──────────────────────────────────────────────


def prepare_classification_data(df: pd.DataFrame):
    """
    Prepara los datos para el modelo de clasificación KNN.
    Variables de entrada: u, g, r, i, z, redshift
    Variable objetivo: class (GALAXY, QSO, STAR)
    """
    features = ["u", "g", "r", "i", "z", "redshift"]
    X = df[features]
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def prepare_regression_data(df: pd.DataFrame):
    """
    Prepara los datos para el modelo de regresión lineal.
    Variables de entrada: u, g, r, i, z
    Variable objetivo: redshift (valor numérico continuo)
    """
    features = ["u", "g", "r", "i", "z"]
    X = df[features]
    y = df["redshift"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def prepare_clustering_data(df: pd.DataFrame):
    """
    Prepara los datos para el modelo de clustering KMeans.
    Variables de entrada: u, g, r, i, z (magnitudes fotométricas)
    No hay variable objetivo — es aprendizaje no supervisado.
    """
    features = ["u", "g", "r", "i", "z"]
    X = df[features]
    y = df["class"]  # Solo para comparación visual

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_test, scaler
