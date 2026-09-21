"""Muestreo estratificado y procesamiento de lotes."""
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


def stratified_subset(split, size, seed):
    # Conserva aproximadamente la proporción original de cada clase
    if size >= len(split):
        return split
    return split.train_test_split(
        train_size=size, stratify_by_column="labels", seed=seed
    )["train"]


def load_splits(config):
    """Carga los splits oficiales y reduce cada uno sin mezclar imágenes."""
    dataset = load_dataset(config.dataset_name)
    train_data = stratified_subset(dataset[config.train_split], config.train_samples, config.seed)
    validation_data = stratified_subset(dataset[config.validation_split], config.validation_samples, config.seed)
    test_data = stratified_subset(dataset[config.test_split], config.test_samples, config.seed)

    # Las clases se obtienen directamente del dataset
    classes = dataset[config.train_split].features["labels"].names
    return train_data, validation_data, test_data, classes


def create_loaders(train_data, validation_data, test_data, processor, batch_size):
    """Preprocesa imágenes al formar cada lote; sólo train se baraja."""
    def collate_fn(samples):
        batch = processor([x["image"].convert("RGB") for x in samples], return_tensors="pt")
        batch["labels"] = torch.tensor([x["labels"] for x in samples])
        return batch

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)

    return train_loader, validation_loader, test_loader
