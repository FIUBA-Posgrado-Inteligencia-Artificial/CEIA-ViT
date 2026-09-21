"""Creación del clasificador y su optimizador para fine-tuning."""
import torch
from transformers import AutoModelForImageClassification


def create_model(config, classes, device):
    """Reemplaza el clasificador y prepara AdamW para fine-tuning."""
    id2label = {index: name for index, name in enumerate(classes)}
    label2id = {name: index for index, name in id2label.items()}

    # Se reemplaza el clasificador original por uno con tres salidas
    model = AutoModelForImageClassification.from_pretrained(
        config.model_name, num_labels=len(classes), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True
    ).to(device)

    # AdamW actualiza los pesos durante el fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    return model, optimizer
