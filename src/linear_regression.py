import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from data_loader import load_dataset, prepare_regression_data


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


def plot_real_vs_predicted(y_test, y_pred, save_path: str):
    """
    Genera y guarda la gráfica de valores reales vs predichos.
    Si el modelo fuera perfecto, todos los puntos estarían sobre la línea roja.
    Puntos alejados de la línea indican errores grandes del modelo.

    Parámetros:
        y_test: Valores reales de redshift.
        y_pred: Valores predichos por el modelo.
        save_path (str): Ruta donde se guarda la imagen.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color="steelblue", label="Predicciones")
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        label="Modelo perfecto",
    )
    plt.xlabel("Redshift Real")
    plt.ylabel("Redshift Predicho")
    plt.title("Regresión Lineal — Real vs Predicho")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Gráfica guardada en: {save_path}")


# ──────────────────────────────────────────────
# modelo
# ──────────────────────────────────────────────


def train_linear_regression(X_train, y_train):
    """
    Entrena el modelo de regresión lineal.
    La regresión lineal predice un valor numérico continuo
    a diferencia de la clasificación que predice categorías.

    Parámetros:
        X_train: Datos de entrenamiento escalados.
        y_train: Valores objetivo de entrenamiento (redshift).

    Retorna:
        model (LinearRegression): Modelo entrenado.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_regression(model, X_test, y_test):
    """
    Evalúa el modelo de regresión con MSE y R².

    MSE (Mean Squared Error): promedio de los errores al cuadrado.
    Elevar al cuadrado evita que errores positivos y negativos se cancelen.

    R² (Coeficiente de determinación): porcentaje de la variación del
    redshift que explica el modelo. R²=1 es perfecto, R²=0 es inútil.

    Parámetros:
        model: Modelo de regresión entrenado.
        X_test: Datos de prueba escalados.
        y_test: Valores reales de redshift.

    Retorna:
        y_pred: Predicciones del modelo.
        mse (float): Mean Squared Error.
        r2 (float): Coeficiente R².
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.4f}")
    print(f"R²:  {r2:.4f}")
    return y_pred, mse, r2


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

    # 1. Cargar y preparar datos
    df = load_dataset("dataset/sdss_sample.csv")
    X_train, X_test, y_train, y_test, scaler = prepare_regression_data(df)

    # 2. Entrenar modelo
    model = train_linear_regression(X_train, y_train)

    # 3. Evaluar modelo
    y_pred, mse, r2 = evaluate_regression(model, X_test, y_test)

    # 4. Guardar métricas
    save_metrics(
        {"MSE": round(mse, 4), "R2": round(r2, 4)},
        "outputs/metrics/linear_regression_metrics.txt",
    )

    # 5. Guardar gráfica
    plot_real_vs_predicted(y_test, y_pred, "outputs/plots/linear_regression.png")

    # 6. Guardar modelo
    save_model(model, "models/linear_regression_model.pkl")
