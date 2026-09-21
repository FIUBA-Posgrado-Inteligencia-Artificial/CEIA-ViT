"""Una época de entrenamiento, independiente de MLflow."""
import torch


def train_epoch(loader, model, device, optimizer):
    """Recorre train y actualiza los pesos; devuelve loss, accuracy y predicciones."""
    # Activa el modo de entrenamiento
    model.train(True)
    loss_sum, correct, total = 0, 0, 0
    labels, predictions = [], []

    for batch in loader:
        # Mueve imágenes y etiquetas a GPU o CPU
        batch = {key: value.to(device) for key, value in batch.items()}

        # Los gradientes sólo son necesarios durante train
        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            output = model(**batch)
            output.loss.backward()
            optimizer.step()

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
