# -*- coding: utf-8 -*-
{% if cookiecutter.interface == "fastapi" %}
from fastapi import FastAPI

app = FastAPI(title="{{ cookiecutter.project_name }}")

@app.get("/")
def root():
    return {"ok": True, "message": "Estructura base generada correctamente", "project": "{{ cookiecutter.project_name }}"}

if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
    except Exception:
        print("FastAPI seleccionado. Instalá: pip install fastapi uvicorn")
        print("Luego ejecutá: uvicorn app.main:app --reload")

{% elif cookiecutter.interface == "streamlit" %}
try:
    import streamlit as st
    st.set_page_config(page_title="{{ cookiecutter.project_name }}", layout="centered")
    st.title("{{ cookiecutter.project_name }}")
    st.write("Estructura base generada (Streamlit).")
    st.caption("Para ejecutar: streamlit run app/main.py")
except Exception:
    print("Streamlit no está instalado. Instalá: pip install streamlit")
    print("Luego ejecutá: streamlit run app/main.py")

{% else %}
def main():
    print("Estructura base generada (CLI) — {{ cookiecutter.project_name }}")

if __name__ == "__main__":
    main()
{% endif %}
