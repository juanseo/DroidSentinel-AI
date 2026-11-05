"""Aplicación web Flask para clasificar perfiles de permisos Android."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "model.joblib"
METADATA_FILE = BASE_DIR / "models" / "metadata.json"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

if not MODEL_FILE.exists() or not METADATA_FILE.exists():
    raise RuntimeError(
        "No se encontró el modelo entrenado. Ejecuta primero: python train_model.py"
    )

# Solo cargar artefactos generados por el propio proyecto y de confianza.
bundle = joblib.load(MODEL_FILE)
model = bundle["pipeline"]
FEATURES = list(bundle["features"])
THRESHOLD = float(bundle.get("threshold", 0.50))
with open(METADATA_FILE, encoding="utf-8") as f:
    metadata = json.load(f)


def classify_permissions(permission_list):
    """Convierte los permisos elegidos en el vector binario esperado por el modelo."""
    if not isinstance(permission_list, list):
        raise ValueError("'permissions' debe ser una lista de nombres de permisos.")
    if any(not isinstance(item, str) for item in permission_list):
        raise ValueError("Cada permiso debe ser una cadena de texto.")

    selected = sorted(set(permission_list))
    unknown = sorted(set(selected) - set(FEATURES))
    if unknown:
        raise ValueError({"message": "Hay permisos que no existen en el dataset.", "unknown": unknown})

    vector = pd.DataFrame(0, index=[0], columns=FEATURES, dtype=np.int8)
    if selected:
        vector.loc[0, selected] = 1

    probabilities = model.predict_proba(vector)[0]
    classes = list(model.classes_)
    malware_probability = float(probabilities[classes.index(1)])
    prediction = int(malware_probability >= THRESHOLD)

    # Importancia global de Gradient Boosting, filtrada a permisos elegidos.
    # No es una explicación causal ni un análisis de la ejecución de la app.
    selected_important = []
    fitted_model = model.named_steps["model"]
    variance_mask = model.named_steps["variance"].get_support()
    effective_features = np.asarray(FEATURES)[variance_mask]
    importances = getattr(fitted_model, "feature_importances_", None)
    if importances is not None:
        importance_map = dict(zip(effective_features, importances))
        selected_important = [
            {"permission": name, "importance": round(float(importance_map[name]), 5)}
            for name in selected
            if name in importance_map and importance_map[name] > 0
        ]
        selected_important.sort(key=lambda item: item["importance"], reverse=True)
        selected_important = selected_important[:5]

    return {
        "prediction": prediction,
        "label": "Malware" if prediction == 1 else "Benigna",
        "malware_probability": round(malware_probability, 4),
        "threshold": THRESHOLD,
        "selected_permissions": selected,
        "top_selected_permissions": selected_important,
        "notice": (
            "Clasificación orientativa basada en permisos declarados. No confirma "
            "malware ni sustituye análisis estático/dinámico del APK."
        ),
    }


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self'; script-src 'self'; "
        "img-src 'self' data:; form-action 'self'; frame-ancestors 'none'"
    )
    return response


@app.route("/", methods=["GET", "POST"])
def index():
    selected = request.form.getlist("permissions") if request.method == "POST" else []
    result = None
    error = None
    if request.method == "POST":
        try:
            result = classify_permissions(selected)
        except ValueError as exc:
            detail = exc.args[0]
            error = detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)

    return render_template(
        "index.html",
        permissions=FEATURES,
        selected=selected,
        result=result,
        error=error,
        metadata=metadata,
    )


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or "permissions" not in payload:
        return jsonify({"error": "Envía JSON con la propiedad 'permissions' como lista."}), 400
    try:
        return jsonify(classify_permissions(payload["permissions"]))
    except ValueError as exc:
        detail = exc.args[0]
        return jsonify({"error": detail}), 400


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": metadata["selected_model"]})


if __name__ == "__main__":
    # debug=False evita exponer el depurador interactivo.
    app.run(host="127.0.0.1", port=5000, debug=False)
