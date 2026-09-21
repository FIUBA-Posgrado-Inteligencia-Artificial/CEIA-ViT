"""Figuras del ejercicio; quien llama decide cuándo mostrarlas o registrarlas."""
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_examples(train_data, classes, num_examples):
    """Muestra las primeras imágenes con sus clases."""
    # Se observan algunas imágenes antes de entrenar
    sample_figure, axes = plt.subplots(1, num_examples, figsize=(10, 3))
    axes = np.atleast_1d(axes)
    for sample, ax in zip(train_data.select(range(num_examples)), axes):
        ax.imshow(sample["image"])
        ax.set_title(classes[sample["labels"]])
        ax.axis("off")
    plt.tight_layout()
    return sample_figure


def class_counts(splits_data, classes):
    """Cuenta ejemplos por clase en cada split."""
    class_distribution = {}
    for split_name, split in splits_data.items():
        counts = Counter(split["labels"])
        class_distribution[split_name] = {
            class_name: counts.get(class_id, 0)
            for class_id, class_name in enumerate(classes)
        }

    return class_distribution


def plot_distribution(class_distribution, classes):
    """Compara la distribución de los tres splits."""
    distribution_figure, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(classes))
    for offset, (name, counts) in zip([-0.25, 0, 0.25], class_distribution.items()):
        values = [counts[class_name] for class_name in classes]
        ax.bar(positions + offset, values, width=0.25, label=name)
    ax.set(xticks=positions, xticklabels=classes, ylabel="Cantidad", title="Distribución por clase")
    ax.legend()
    distribution_figure.tight_layout()
    return distribution_figure


def plot_confusion(y_true, y_pred, classes):
    """Construye la matriz de confusión de test."""
    confusion_figure, ax = plt.subplots(figsize=(7, 5))
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set(xlabel="Predicción", ylabel="Clase real")
    confusion_figure.tight_layout()

    return confusion_figure


def plot_learning_curves(history):
    """Grafica loss y accuracy de train, validation y test por época."""
    curve_figure, curve_axes = plt.subplots(1, 2, figsize=(10, 4))
    curve_axes[0].plot(history["train_loss"], marker="o", label="Train")
    curve_axes[0].plot(history["validation_loss"], marker="o", label="Validation")
    curve_axes[0].plot(history["test_loss"], marker="o", linestyle="--", label="Test")
    curve_axes[0].set(title="Loss por época", xlabel="Época")
    curve_axes[1].plot(history["train_accuracy"], marker="o", label="Train")
    curve_axes[1].plot(history["validation_accuracy"], marker="o", label="Validation")
    curve_axes[1].plot(history["test_accuracy"], marker="o", linestyle="--", label="Test")
    curve_axes[1].set(title="Accuracy por época", xlabel="Época", ylim=(0, 1))
    for axis in curve_axes:
        axis.legend()
    curve_figure.tight_layout()

    return curve_figure


def plot_comparison(train_metrics, validation_metrics, test_metrics):
    """Compara las métricas finales del mejor modelo."""
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    comparison_figure, (loss_ax, score_ax) = plt.subplots(1, 2, figsize=(13, 4))
    splits = [("Train", train_metrics), ("Validation", validation_metrics), ("Test", test_metrics)]
    loss_ax.bar([name for name, _ in splits], [values[f"{name.lower()}_loss"] for name, values in splits])
    loss_ax.set_title("Loss final")
    positions = list(range(len(metric_names)))
    for offset, (name, values) in zip([-0.25, 0, 0.25], splits):
        score_ax.bar([x + offset for x in positions],
                     [values[f"{name.lower()}_{metric}"] for metric in metric_names],
                     width=0.25, label=name)
    score_ax.set(xticks=positions, xticklabels=metric_names, ylim=(0, 1), ylabel="Valor")
    score_ax.tick_params(axis="x", rotation=25)
    score_ax.set_title("Métricas finales")
    score_ax.legend()
    comparison_figure.tight_layout()

    return comparison_figure
