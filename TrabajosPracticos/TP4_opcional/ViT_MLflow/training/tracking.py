"""Orquesta la corrida y conserva métricas, artefactos y mejor modelo."""
import copy
import tempfile

import mlflow
from sklearn.metrics import classification_report

from src.metrics import calculate_metrics
from inference.evaluation import evaluate
from .engine import train_epoch
from src.visualization import plot_confusion, plot_learning_curves, plot_comparison


def run_experiment(
    config, model, processor, optimizer, device, train_loader,
    validation_loader, test_loader, classes, sample_figure,
    distribution_figure, class_distribution,
):
    """Entrena, selecciona por validation y registra los resultados originales en MLflow."""
    mlflow.set_experiment(config.experiment_name)

    # MLflow registra CPU, memoria y GPU durante la ejecución
    with mlflow.start_run(run_name=config.run_name, log_system_metrics=True):
        # Parámetros y etiquetas de la corrida
        mlflow.log_params({
            "model": config.model_name, "dataset": config.dataset_name,
            "epochs": config.epochs, "batch_size": config.batch_size,
            "learning_rate": config.learning_rate, "train_samples": config.train_samples,
            "validation_samples": config.validation_samples, "test_samples": config.test_samples,
            "seed": config.seed,
            "metric_average": config.metric_average,
            "weighted_average": config.weighted_average,
        })
        mlflow.set_tags({"task": "image-classification", "architecture": "ViT"})

        # Artefactos iniciales
        mlflow.log_figure(sample_figure, "dataset/examples.png")
        mlflow.log_figure(distribution_figure, "dataset/class_distribution.png")
        mlflow.log_dict({"classes": classes}, "dataset/classes.json")
        mlflow.log_dict(class_distribution, "dataset/class_distribution.json")
        mlflow.log_dict(model.config.to_dict(), "model/config.json")

        # Train y validation se comparan después de cada época
        history = {
            "train_loss": [], "validation_loss": [], "test_loss": [],
            "train_accuracy": [], "validation_accuracy": [], "test_accuracy": [],
        }
        best_validation_loss = float("inf")
        best_state = None
        best_epoch = 0
        for epoch in range(config.epochs):
            train_epoch(train_loader, model, device, optimizer)
            train_metrics, _, _ = calculate_metrics(
                "train", evaluate(train_loader, model, device),
                config.metric_average, config.weighted_average,
            )
            validation_metrics, _, _ = calculate_metrics(
                "validation", evaluate(validation_loader, model, device),
                config.metric_average, config.weighted_average,
            )
            test_epoch_metrics, _, _ = calculate_metrics(
                "test", evaluate(test_loader, model, device),
                config.metric_average, config.weighted_average,
            )
            epoch_metrics = {**train_metrics, **validation_metrics, **test_epoch_metrics}
            mlflow.log_metrics(epoch_metrics, step=epoch)
            for name in history:
                history[name].append(epoch_metrics[name])
            # Conserva el modelo con menor loss de validation
            if validation_metrics["validation_loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["validation_loss"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
            print(f"Época {epoch + 1}: train_loss={train_metrics['train_loss']:.3f} - "
                  f"validation_loss={validation_metrics['validation_loss']:.3f}")

        # Restaura el mejor modelo antes de la evaluación final
        model.load_state_dict(best_state)
        train_metrics, _, _ = calculate_metrics(
            "train", evaluate(train_loader, model, device),
            config.metric_average, config.weighted_average,
        )
        validation_metrics, _, _ = calculate_metrics(
            "validation", evaluate(validation_loader, model, device),
            config.metric_average, config.weighted_average,
        )
        mlflow.log_metrics({**train_metrics, **validation_metrics}, step=config.epochs)
        mlflow.log_param("best_epoch", best_epoch)

        # Evaluación final de test con el mejor modelo restaurado
        test_metrics, y_true, y_pred = calculate_metrics(
            "test", evaluate(test_loader, model, device),
            config.metric_average, config.weighted_average,
        )
        mlflow.log_metrics(test_metrics, step=config.epochs)
        report = classification_report(
            y_true, y_pred, labels=range(len(classes)), target_names=classes,
            output_dict=True, zero_division=0
        )

        # Matriz de confusión del conjunto de test
        confusion_figure = plot_confusion(y_true, y_pred, classes)

        # Curvas para observar aprendizaje y posible overfitting
        curve_figure = plot_learning_curves(history)

        # Comparación final de train, validation y test
        comparison_figure = plot_comparison(train_metrics, validation_metrics, test_metrics)

        # Los gráficos y reportes quedan disponibles como artefactos
        mlflow.log_figure(curve_figure, "results/learning_curves.png")
        mlflow.log_figure(confusion_figure, "results/confusion_matrix.png")
        mlflow.log_figure(comparison_figure, "results/train_validation_test.png")
        mlflow.log_dict({"train": train_metrics, "validation": validation_metrics,
                         "test": test_metrics}, "results/metrics.json")
        mlflow.log_dict(report, "results/classification_report.json")
        mlflow.log_text(f"Fine-tuning de {config.model_name} sobre {config.dataset_name}.", "results/summary.txt")

        # Guarda el mejor modelo y su procesador como artefactos
        with tempfile.TemporaryDirectory() as model_dir:
            model.save_pretrained(model_dir)
            processor.save_pretrained(model_dir)
            mlflow.log_artifacts(model_dir, artifact_path="model")

    return {
        "history": history, "best_epoch": best_epoch,
        "train_metrics": train_metrics, "validation_metrics": validation_metrics,
        "test_metrics": test_metrics, "y_true": y_true, "y_pred": y_pred,
        "report": report, "curve_figure": curve_figure,
        "comparison_figure": comparison_figure, "confusion_figure": confusion_figure,
    }
