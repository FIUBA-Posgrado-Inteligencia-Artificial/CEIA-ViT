"""Reproducibilidad, dispositivo y modelo preentrenado."""
import random

import numpy as np
import torch
from transformers import AutoImageProcessor


def setup_device(seed):
    """Inicializa las semillas para reproducibilidad y elige GPU o CPU."""
    # Las semillas permiten repetir el experimento
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def create_processor(model_name):
    """Carga el tamaño y la normalización esperados por el ViT."""
    return AutoImageProcessor.from_pretrained(model_name)
