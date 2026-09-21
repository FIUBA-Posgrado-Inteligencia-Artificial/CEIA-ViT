"""Métricas de clasificación con los promedios del ejercicio."""
from sklearn.metrics import precision_recall_fscore_support


def calculate_metrics(split, results, metric_average="macro", weighted_average="weighted"):
    """Devuelve métricas, etiquetas y predicciones para un split."""
    # Calcula las mismas métricas para cada split
    loss, accuracy, labels, predictions = results
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=metric_average, zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average=weighted_average, zero_division=0
    )
    metrics = {
        f"{split}_loss": float(loss),
        f"{split}_accuracy": float(accuracy),
        f"{split}_precision_macro": float(precision),
        f"{split}_recall_macro": float(recall),
        f"{split}_f1_macro": float(f1),
        f"{split}_f1_weighted": float(f1_weighted),
    }
    return metrics, labels, predictions
