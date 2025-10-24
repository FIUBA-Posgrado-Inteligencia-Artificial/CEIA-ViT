# -*- coding: utf-8 -*-

from fastapi import FastAPI

app = FastAPI(title="ejemplo_de_estructura")

@app.get("/")
def root():
    return {"ok": True, "message": "Estructura base generada correctamente", "project": "ejemplo_de_estructura"}

if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
    except Exception:
        print("FastAPI seleccionado. Instalá: pip install fastapi uvicorn")
        print("Luego ejecutá: uvicorn app.main:app --reload")


