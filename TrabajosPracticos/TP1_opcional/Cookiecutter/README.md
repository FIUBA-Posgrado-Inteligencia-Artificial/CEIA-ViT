# Plantilla Cookiecutter para proyectos ViT (Vision Transformer)

Esta plantilla permite crear, de forma rápida y ordenada, la estructura base de un proyecto orientado a entrenamiento o fine-tuning de modelos **Vision Transformer (ViT)** en PyTorch.
Su propósito es que no tengas que definir manualmente la jerarquía de carpetas cada vez que inicies un nuevo experimento.

---

## ¿Qué hace esta plantilla?

Al ejecutar `cookiecutter`, la plantilla te solicita algunas opciones básicas (por ejemplo, el nombre del proyecto o el tipo de interfaz que querés usar).
Con esas respuestas, genera **únicamente la estructura de carpetas y archivos mínimos**, dejando listo un esqueleto de proyecto limpio y escalable.

Esto asegura que todos los proyectos de entrenamiento o inferencia con ViT tengan un mismo orden lógico desde el inicio.

---

## Cómo usarla

1. Cloná o descargá este repositorio.
2. Desde una terminal, ubicándote en la carpeta que contiene `cookiecutter-vit`, ejecutá:

   ```bash
   cookiecutter cookiecutter-vit
   ```
3. Respondé las preguntas (nombre del proyecto, interfaz, habilitar tests o ejemplos, etc.).
4. Se creará una nueva carpeta con el nombre del proyecto y toda la estructura base.

---

## Estructura resultante

El árbol exacto puede variar según las opciones elegidas, pero en general se verá así:

```
{{ project_name }}/
├─ app/                        # Punto de entrada (FastAPI, Streamlit o CLI)
│  ├─ __init__.py
│  └─ main.py
│
├─ models/                     # Configuraciones o pesos del modelo
│  └─ config.json
│
├─ configs/                    # Archivos de configuración
│  ├─ envs/
│  │  ├─ dev/
│  │  ├─ staging/
│  │  └─ prod/
│  └─ model/
│
├─ data/                       # Estructura de datos del proyecto
│  ├─ raw/                     # Datos originales
│  ├─ interim/                 # Datos intermedios o preprocesados
│  └─ processed/               # Datos listos para entrenamiento
│
├─ docs/                       # Documentación técnica
│  ├─ architecture/
│  ├─ decisions/
│  └─ api/
│
├─ notebooks/                  # Jupyter Notebooks para exploración o pruebas
├─ scripts/                    # Scripts auxiliares (entrenamiento, evaluación, etc.)
├─ reports/                    # Resultados de experimentos
├─ logs/                       # Logs locales de ejecución
│
├─ src/                        # Código fuente principal
│  └─ vit/
│     ├─ data/                 # Cargadores y preprocesamiento de datasets
│     ├─ models/               # Arquitecturas (timm, torchvision, custom)
│     ├─ train/                # Lógica de entrenamiento y validación
│     ├─ eval/                 # Evaluación de modelos y métricas
│     ├─ inference/            # Scripts de inferencia
│     ├─ transforms/           # Augmentations y normalizaciones
│     └─ utils/                # Funciones auxiliares (logging, config, etc.)
│
├─ tests/                      # (opcional) Tests unitarios e integrales
│  ├─ unit/
│  ├─ integration/
│  └─ e2e/
│
├─ examples/                   # (opcional) Ejemplos y plantillas de uso
│  ├─ quickstarts/
│  └─ templates/
│
├─ checkpoints/                # Modelos guardados o mejores pesos
├─ .github/workflows/          # Pipelines de CI/CD
├─ requirements.txt            # Dependencias básicas (PyTorch, timm, etc.)
└─ README.md                   # Descripción del proyecto
```

---

## Resumen

* Las carpetas base (`src/`, `configs/`, `data/`, `docs/`, `scripts/`, etc.) se crean siempre.
* Las carpetas `tests/` y `examples/` se generan solo si las activás durante la ejecución.
* El propósito es ofrecer una **base uniforme** para proyectos de visión, adaptable a distintos pipelines o experimentos.

---

## Detalle de los hooks

**pre_gen_project.py**
Verifica combinaciones válidas y asegura que los parámetros básicos estén bien definidos (por ejemplo, interfaz válida).

**post_gen_project.py**
Crea las carpetas condicionalmente según tus elecciones, agrega `.gitkeep` donde sea necesario y genera los archivos semilla (`README.md`, `requirements.txt`, etc.).

---

## Filosofía

* Generar **solo lo necesario**: sin código de entrenamiento, sin dependencias impuestas.
* Estructura **modular y extensible**: podés agregar tus propios módulos, notebooks o configuraciones sin romper la organización base.
* Compatible con entornos reproducibles: `requirements.txt` listo para instalar PyTorch, timm y dependencias comunes de visión.

---

## Conclusión

Esta plantilla no busca definir un flujo de trabajo completo, sino brindar un **punto de partida coherente y reutilizable** para cualquier proyecto de Vision Transformer.
Con un par de preguntas iniciales, tenés un esqueleto limpio, reproducible y preparado para crecer con tu proyecto.
