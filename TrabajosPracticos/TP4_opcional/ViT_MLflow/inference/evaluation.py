"""Inferencia sobre lotes etiquetados para evaluar el clasificador."""
import torch


def evaluate(loader, model, device):
    """Devuelve loss, accuracy, etiquetas y predicciones sin calcular gradientes."""
    # Desactiva dropout y otros comportamientos propios del entrenamiento
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    labels, predictions = [], []

    for batch in loader:
        # Mueve imágenes y etiquetas a GPU o CPU
        batch = {key: value.to(device) for key, value in batch.items()}

        # La evaluación no necesita optimizador ni gradientes
        with torch.no_grad():
            output = model(**batch)

        # Clase con mayor puntuación para cada imagen
        predicted = output.logits.argmax(1)
        batch_size = batch["labels"].size(0)
        loss_sum += output.loss.item() * batch_size
        correct += (predicted == batch["labels"]).sum().item()
        total += batch_size
        labels.extend(batch["labels"].cpu().tolist())
        predictions.extend(predicted.cpu().tolist())

    # Promedios y valores necesarios para calcular otras métricas
    return loss_sum / total, correct / total, labels, predictions
