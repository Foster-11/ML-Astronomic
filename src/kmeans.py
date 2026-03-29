import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from data_loader import load_dataset, prepare_clustering_data


# ──────────────────────────────────────────────
# Funciones auxiliares
# ──────────────────────────────────────────────


def plot_clusters_vs_real(X_test, clusters, y_test, save_path: str):
    """
    Genera y guarda una comparación visual entre los clusters de KMeans
    y las clases reales del dataset (GALAXY, QSO, STAR).

    Si KMeans hace un buen trabajo, los patrones de colores de ambas
    gráficas deberían parecerse — indicando que las magnitudes fotométricas
    contienen información suficiente para distinguir los objetos estelares.

    Parámetros:
        X_test: Datos de prueba escalados.
        clusters: Etiquetas de cluster asignadas por KMeans (0, 1, 2).
        y_test: Clases reales (GALAXY, QSO, STAR) — solo para comparar.
        save_path (str): Ruta donde se guarda la imagen.
    """
    # Codificar clases reales a números para graficar con colores
    y_encoded = LabelEncoder().fit_transform(y_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfica 1 — Clusters encontrados por KMeans (sin conocer etiquetas)
    ax1.scatter(X_test[:, 0], X_test[:, 1], c=clusters, cmap="viridis", alpha=0.5)
    ax1.set_title("Clusters KMeans (no supervisado)")
    ax1.set_xlabel("u (magnitud fotométrica)")
    ax1.set_ylabel("g (magnitud fotométrica)")

    # Gráfica 2 — Clases reales del dataset
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_encoded, cmap="viridis", alpha=0.5)
    ax2.set_title("Clases Reales (GALAXY / QSO / STAR)")
    ax2.set_xlabel("u (magnitud fotométrica)")
    ax2.set_ylabel("g (magnitud fotométrica)")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Gráfica guardada en: {save_path}")


# ──────────────────────────────────────────────
# Funciones del modelo
# ──────────────────────────────────────────────


def train_kmeans(X_train, n_clusters: int = 3):
    """
    Entrena el modelo KMeans con el número de clusters indicado.

    A diferencia de KNN y regresión lineal, KMeans es aprendizaje
    NO SUPERVISADO — no recibe etiquetas (y) durante el entrenamiento.
    Solo agrupa por similitud en los valores de las variables de entrada.

    El algoritmo:
    1. Coloca n_clusters centroides aleatorios
    2. Asigna cada punto al centroide más cercano
    3. Recalcula los centroides
    4. Repite hasta converger

    Parámetros:
        X_train: Datos de entrenamiento escalados (sin etiquetas).
        n_clusters (int): Número de grupos a formar (default=3).

    Retorna:
        kmeans (KMeans): Modelo entrenado.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train)
    return kmeans


def predict_clusters(kmeans, X_test):
    """
    Asigna cada objeto del conjunto de prueba a un cluster.
    Los clusters se identifican con números (0, 1, 2) — no con nombres.

    Parámetros:
        kmeans: Modelo KMeans entrenado.
        X_test: Datos de prueba escalados.

    Retorna:
        clusters: Array con el número de cluster asignado a cada objeto.
    """
    clusters = kmeans.predict(X_test)
    print(f"Clusters únicos encontrados: {set(clusters)}")
    return clusters


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
# Ejecucion principal
# ──────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Cargar y preparar datos
    df = load_dataset("dataset/sdss_sample.csv")
    X_train, X_test, y_test, scaler = prepare_clustering_data(df)

    # 2. Entrenar KMeans con 3 clusters
    kmeans = train_kmeans(X_train, n_clusters=3)

    # 3. Predecir clusters en el conjunto de prueba
    clusters = predict_clusters(kmeans, X_test)

    # 4. Guardar gráfica comparativa clusters vs clases reales
    plot_clusters_vs_real(
        X_test, clusters, y_test, save_path="outputs/plots/kmeans_clusters.png"
    )

    # 5. Guardar modelo
    save_model(kmeans, "models/kmeans_model.pkl")
