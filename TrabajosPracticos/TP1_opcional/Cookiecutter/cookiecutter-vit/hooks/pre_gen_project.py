# -*- coding: utf-8 -*-
"""
Validaciones mínimas para estructura ViT (solo carpetas).
"""
import sys

INTERFACE = "{{ cookiecutter.interface }}"

def fail(msg: str):
    print(f"[cookiecutter][pre_gen] ERROR: {msg}")
    sys.exit(1)

if INTERFACE not in {"none", "fastapi", "streamlit"}:
    fail(f"interface inválida: {INTERFACE}")

print("[cookiecutter][pre_gen] OK (estructura).")
