"""Lectura de parámetros externos, sin valores duplicados en el código."""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Estructura de los parámetros definidos en config/parameters.json."""
    model_name: str
    dataset_name: str
    train_split: str
    validation_split: str
    test_split: str
    train_samples: int
    validation_samples: int
    test_samples: int
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    metric_average: str
    weighted_average: str
    experiment_name: str
    run_name: str
    num_examples: int


def load_config(path: str | Path) -> ExperimentConfig:
    """Carga un JSON; todas las claves de ExperimentConfig son obligatorias.

    Las rutas relativas se resuelven desde el directorio de trabajo actual.
    Las claves faltantes o desconocidas producen un error al crear la configuración.
    """
    with Path(path).open(encoding="utf-8") as config_file:
        parameters = json.load(config_file)
    return ExperimentConfig(**parameters)
