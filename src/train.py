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
from sklearn.model_selection import GridSearchCV, KFold

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
    base = spec["build"]()
    X_sel = X[spec["numeric"] + ["Species"]]
    grid = spec.get("param_grid")

    t0 = time.time()
    if grid:
        # Вложенная проверка: внутри каждого внешнего фолда параметры подбираются
        # только по его обучающей части, поэтому оценка честная.
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        model = GridSearchCV(base, grid, cv=inner_cv, scoring="neg_mean_squared_error", n_jobs=-1
        )
    else:
        model = base

    metrics = cv_metrics(model, X_sel, y, seed)
    oof = oof_predictions(model, X_sel, y, seed)
    extra_params = {}
    if grid:
        # Какие параметры выбрались на всей обучающей части (для отчёта и сохранения модели)
        model.fit(X_sel, y)
        extra_params = {
            "param_grid": str(grid),
            "best_params": str(model.best_params_),
            "inner_cv": "KFold(5, shuffle)",
        }
        print(f"   лучшие параметры: {model.best_params_}")
    metrics["duration_sec"] = round(time.time() - t0, 1)

    oof_path = Path("runs") / f"oof_{name}.csv"
    oof_path.parent.mkdir(exist_ok=True)
    oof.to_csv(oof_path)

    if use_mlflow:
        import mlflow

        saved = load_split()
        params = {
            "experiment": name,
            "description": spec["description"],
            "model": type(base).__name__,
            "features": ",".join(X_sel.columns),
            "seed": seed,
            "cv": "RepeatedKFold(5 folds x 3 repeats)",
            "n_train": len(y),
            "data_sha256": saved["data_sha256"],
            **extra_params,
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
    # Накопительная таблица: запуск части экспериментов не стирает результаты остальных
    results_path = Path("runs/cv_results.csv")
    if results_path.exists():
        old = pd.read_csv(results_path, index_col="experiment")
        merged = pd.concat([old.drop(index=table.index, errors="ignore"), table])
        merged = merged.reindex([n for n in EXPERIMENTS if n in merged.index])
    else:
        merged = table
    merged.to_csv(results_path)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
