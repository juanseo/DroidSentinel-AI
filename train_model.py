"""Entrena y selecciona un clasificador para permisos Android.

Uso:
    python train_model.py --csv train.csv

La validación agrupa perfiles de permisos idénticos para impedir que copias del
mismo perfil aparezcan simultáneamente en entrenamiento y validación. Los
perfiles idénticos con etiquetas contradictorias se excluyen del entrenamiento.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TARGET = "type"


def load_and_clean(csv_path: Path):
    """Lee el CSV, valida sus columnas y elimina perfiles de etiqueta ambigua."""
    df = pd.read_csv(csv_path, sep=";")
    if TARGET not in df.columns:
        raise ValueError(f"No existe la columna objetivo obligatoria '{TARGET}'.")

    y = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError("La columna 'type' debe contener exactamente las etiquetas 0 y 1.")

    X = df.drop(columns=[TARGET]).copy()
    if X.empty:
        raise ValueError("No se encontraron columnas de permisos.")
    if X.columns.duplicated().any():
        raise ValueError("El CSV contiene nombres de permisos duplicados.")

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="raise")
    if X.isna().any().any():
        raise ValueError("Hay valores nulos en los permisos. Limpia el CSV antes de entrenar.")
    values = set(np.unique(X.to_numpy()))
    if not values.issubset({0, 1}):
        raise ValueError(f"Los permisos deben ser binarios (0/1); se encontraron: {values}.")

    # El hash representa el vector completo de permisos, independientemente del índice.
    groups = pd.util.hash_pandas_object(X, index=False).astype("uint64")
    label_counts = pd.DataFrame({"group": groups, "target": y}).groupby("group")["target"].nunique()
    ambiguous_groups = set(label_counts[label_counts > 1].index)
    keep = ~groups.isin(ambiguous_groups)

    removed_rows = int((~keep).sum())
    X = X.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)
    groups = groups.loc[keep].reset_index(drop=True)

    if y.nunique() != 2 or groups.nunique() < 4:
        raise ValueError("No hay suficientes perfiles consistentes para validar el modelo.")

    return X, y, groups, {
        "original_rows": int(len(df)),
        "training_rows": int(len(y)),
        "permission_columns": int(X.shape[1]),
        "variable_permissions": int((X.nunique() > 1).sum()),
        "constant_permissions": int((X.nunique() <= 1).sum()),
        "unique_permission_profiles": int(groups.nunique()),
        "ambiguous_profiles_removed": int(len(ambiguous_groups)),
        "ambiguous_rows_removed": removed_rows,
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
    }


def build_candidates() -> dict[str, Any]:
    """Modelos comparados con la misma validación agrupada."""
    return {
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
        "Regresión logística": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            C=1.0,
            random_state=42,
        ),
        "SVM RBF": SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
        "Naive Bayes Bernoulli": BernoulliNB(alpha=1.0),
    }


def evaluate_candidates(X, y, groups, candidates):
    """Evalúa con 3 repeticiones de validación cruzada estratificada de 4 grupos."""
    all_results: dict[str, dict[str, float]] = {}
    metric_names = [
        "accuracy", "precision", "recall", "f1", "f2",
        "balanced_accuracy", "roc_auc",
    ]

    for name, estimator in candidates.items():
        fold_metrics = []
        for seed in range(3):
            cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
            for train_idx, test_idx in cv.split(X, y, groups):
                pipeline = Pipeline([
                    ("variance", VarianceThreshold(threshold=0.0)),
                    ("model", estimator),
                ])
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)
                scores = pipeline.predict_proba(X_test)[:, 1]
                fold_metrics.append([
                    accuracy_score(y_test, predictions),
                    precision_score(y_test, predictions, zero_division=0),
                    recall_score(y_test, predictions, zero_division=0),
                    f1_score(y_test, predictions, zero_division=0),
                    fbeta_score(y_test, predictions, beta=2, zero_division=0),
                    balanced_accuracy_score(y_test, predictions),
                    roc_auc_score(y_test, scores),
                ])

        averages = np.asarray(fold_metrics).mean(axis=0)
        all_results[name] = {
            metric: round(float(value), 4)
            for metric, value in zip(metric_names, averages)
        }
        print(
            f"{name:24s} | F2={averages[4]:.3f} | F1={averages[3]:.3f} "
            f"| Recall={averages[2]:.3f} | ROC-AUC={averages[6]:.3f}"
        )

    # En detección de malware priorizamos F2: da más peso al recall y reduce
    # el coste de dejar pasar una aplicación maliciosa (falso negativo).
    selected_name = max(
        all_results,
        key=lambda name: (
            all_results[name]["f2"],
            all_results[name]["recall"],
            all_results[name]["f1"],
        ),
    )
    return selected_name, all_results


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de detector de malware Android")
    parser.add_argument("--csv", default=str(BASE_DIR / "train.csv"), help="Ruta al train.csv")
    args = parser.parse_args()
    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {csv_path}")

    print(f"Leyendo: {csv_path}")
    X, y, groups, data_info = load_and_clean(csv_path)
    print("\nResumen de limpieza:")
    print(json.dumps(data_info, indent=2, ensure_ascii=False))

    print("\nEvaluación de modelos (validación agrupada):")
    candidates = build_candidates()
    best_name, cv_results = evaluate_candidates(X, y, groups, candidates)
    best_estimator = candidates[best_name]

    final_pipeline = Pipeline([
        ("variance", VarianceThreshold(threshold=0.0)),
        ("model", best_estimator),
    ])
    final_pipeline.fit(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": final_pipeline,
        "features": list(X.columns),
        "target": TARGET,
        "positive_class": 1,
        "threshold": 0.50,
    }
    joblib.dump(bundle, MODEL_DIR / "model.joblib")

    metadata = {
        **data_info,
        "selected_model": best_name,
        "selection_metric": "F2 (prioriza recall de malware)",
        "decision_threshold": 0.50,
        "cross_validation": {
            "method": "StratifiedGroupKFold",
            "folds": 4,
            "repeats": 3,
            "grouping": "perfiles de permisos idénticos",
        },
        "model_comparison": cv_results,
        "sklearn_version": sklearn.__version__,
        "note": (
            "Métricas de validación cruzada agrupada; no equivalen a una garantía "
            "de rendimiento en aplicaciones nuevas o actuales."
        ),
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nModelo seleccionado: {best_name}")
    print(f"Artefactos guardados en: {MODEL_DIR}")
    print("Ejecuta la aplicación con: python app.py")


if __name__ == "__main__":
    main()
