import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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


def prepare_classification_data(df: pd.DataFrame):
    """
    Prepara los datos para el modelo de clasificación KNN.
    Variables de entrada: u, g, r, i, z, redshift
    Variable objetivo: class (GALAXY, QSO, STAR)

    Parámetros:
        df (pd.DataFrame): Dataset completo.

    Retorna:
        X_train, X_test, y_train, y_test: Conjuntos de entrenamiento y prueba escalados.
        scaler (MinMaxScaler): Scaler entrenado (necesario para guardar o reutilizar).
    """
    features = ["u", "g", "r", "i", "z", "redshift"]
    X = df[features]
    y = df["class"]

    # División 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Escalado — fit solo con train para evitar data leakage
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

    Parámetros:
        df (pd.DataFrame): Dataset completo.

    Retorna:
        X_train, X_test, y_train, y_test: Conjuntos de entrenamiento y prueba escalados.
        scaler (MinMaxScaler): Scaler entrenado.
    """
    features = ["u", "g", "r", "i", "z"]
    X = df[features]
    y = df["redshift"]

    # División 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Escalado — fit solo con train para evitar data leakage
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

    Parámetros:
        df (pd.DataFrame): Dataset completo.

    Retorna:
        X_train, X_test: Conjuntos escalados.
        y_test: Clases reales (solo para comparar visualmente con los clusters).
        scaler (MinMaxScaler): Scaler entrenado.
    """
    features = ["u", "g", "r", "i", "z"]
    X = df[features]
    y = df["class"]  # Solo para comparación visual, no para entrenar

    # División 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Escalado — fit solo con train para evitar data leakage
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_test, scaler
