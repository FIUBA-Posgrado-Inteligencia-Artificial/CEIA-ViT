# TP4 — ViT con MLflow

Abrir `TP4_ViT_MLflow.ipynb` con el directorio de trabajo en esta carpeta y ejecutar
las celdas en orden. La primera celda instala las dependencias y el paquete local
con `%pip install -q -e .`. Se necesita acceso a Hugging Face para descargar Beans
y DeiT-Tiny; se usa CUDA si está disponible y CPU en caso contrario.

La notebook consume los módulos de tres carpetas hermanas:

- `src/`: configuración, semillas, dispositivo, procesador, métricas y gráficos.
- `training/`: datos y lotes, creación del clasificador, entrenamiento y MLflow.
- `inference/`: predicciones y evaluación sobre lotes etiquetados, sin gradientes.

Los parámetros se editan en `config/parameters.json`. La notebook los carga con
`from src.config import load_config` y `load_config("config/parameters.json")`.
Para aplicar cambios, ejecutar las celdas desde la carga de configuración.

El entrenamiento se importa con `from training.tracking import run_experiment`.
Este módulo usa `train_epoch` de `training/engine.py` y `evaluate` de
`inference/evaluation.py`. Las funciones también se pueden importar directamente
desde una notebook para estudiar cada etapa. No se ejecuta código al importarlas.

`pyproject.toml` instala `src`, `training` e `inference` en modo editable.

Para consultar las corridas, ejecutar desde `ViT_MLflow/`:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

