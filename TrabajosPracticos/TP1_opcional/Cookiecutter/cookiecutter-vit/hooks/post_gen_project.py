# -*- coding: utf-8 -*-
"""
Crea la estructura de carpetas para un proyecto ViT (sin código).
"""

from pathlib import Path

PROJECT_NAME = "{{ cookiecutter.project_name }}"
INTERFACE = "{{ cookiecutter.interface }}"
ENABLE_TESTS = "{{ cookiecutter.enable_tests }}"
ADD_EXAMPLES = "{{ cookiecutter.add_examples }}"

ROOT = Path(".")

def mk(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def keep(p: Path):
    (p / ".gitkeep").write_text("", encoding="utf-8")

def main():
    base_dirs = [
        ".github/workflows",
        "configs/envs/dev",
        "configs/envs/staging",
        "configs/envs/prod",
        "configs/model",
        "checkpoints",
        "data/raw",
        "data/interim",
        "data/processed",
        "docs",
        "logs",
        "notebooks",
        "reports",
        "scripts",
        "experiments",
        "src/vit/data",
        "src/vit/models",
        "src/vit/train",
        "src/vit/eval",
        "src/vit/inference",
        "src/vit/transforms",
        "src/vit/utils"
    ]

    # interface (solo carpeta)
    if INTERFACE == "fastapi":
        base_dirs.append("src/interfaces/fastapi_app")
    elif INTERFACE == "streamlit":
        base_dirs.append("src/interfaces/streamlit_app")
    else:
        base_dirs.append("src/interfaces/cli")

    if ENABLE_TESTS == "yes":
        base_dirs += ["tests/unit", "tests/integration", "tests/e2e"]

    if ADD_EXAMPLES == "yes":
        base_dirs += ["examples/quickstarts", "examples/templates"]

    # crear
    for d in base_dirs:
        p = ROOT / d
        mk(p)
        keep(p)

    # semillas mínimas
    (ROOT / "README.md").write_text(
        f"# {PROJECT_NAME}\n\nEstructura base para proyecto ViT (solo carpetas).\n",
        encoding="utf-8"
    )
    (ROOT / "requirements.txt").write_text(
        "torch\ntorchvision\ntimm\nnumpy\nscikit-learn\nPillow\n",
        encoding="utf-8"
    )

    print("[cookiecutter][post_gen] Directorios creados:")
    for d in base_dirs:
        print(f" - {d}")

if __name__ == "__main__":
    main()
