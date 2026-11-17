"""Запуск экспериментов и запись результатов в MLflow.

    python -m src.train                    # все эксперименты
    python -m src.train E0-mean E1a-linear-raw
    python -m src.train --no-mlflow        # без записи в MLflow

На этом этапе используется ТОЛЬКО обучающая часть (кросс-валидация).
Тестовая выборка не открывается.
"""
import argparse
import platform
import subprocess
import time
from pathlib import Path

import pandas as pd
import sklearn

from src.data import load_train_test
from src.evaluate import cv_metrics, oof_predictions
from src.experiments import EXPERIMENTS
from src.split import load_config, load_split

TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT = "fish-weight"


def git_info():
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], text=True).strip()
        except Exception:
            return "unknown"

    return {
        "git_commit": run("rev-parse", "--short", "HEAD"),
        "git_dirty": str(bool(run("status", "--porcelain"))),
    }


def run_one(name, spec, X, y, seed, use_mlflow):
    model = spec["build"]()
    X_sel = X[spec["numeric"] + ["Species"]]

    metrics = cv_metrics(model, X_sel, y, seed)
    oof = oof_predictions(model, X_sel, y, seed)
    oof_path = Path("runs") / f"oof_{name}.csv"
    oof_path.parent.mkdir(exist_ok=True)
    oof.to_csv(oof_path)

    if use_mlflow:
        import mlflow

        saved = load_split()
        params = {
            "experiment": name,
            "description": spec["description"],
            "model": type(model).__name__,
            "features": ",".join(X_sel.columns),
            "seed": seed,
            "cv": "RepeatedKFold(5 folds x 3 repeats)",
            "n_train": len(y),
            "data_sha256": saved["data_sha256"],
        }
        tags = {
            **git_info(),
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
        }
        run_name = f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tags(tags)
            mlflow.log_artifact(str(oof_path))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="имена экспериментов (по умолчанию все)")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    names = args.names or list(EXPERIMENTS)
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Нет такого эксперимента: {unknown}. Доступны: {list(EXPERIMENTS)}")

    seed = load_config()["seed"]
    X_train, y_train, _, _ = load_train_test()  # тест здесь не используется

    use_mlflow = not args.no_mlflow
    if use_mlflow:
        import mlflow

        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

    rows = []
    for name in names:
        print(f"Запуск {name} ...")
        m = run_one(name, EXPERIMENTS[name], X_train, y_train, seed, use_mlflow)
        rows.append({"experiment": name, **m})

    table = pd.DataFrame(rows).set_index("experiment")
    Path("runs").mkdir(exist_ok=True)
    table.to_csv("runs/cv_results.csv")
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
