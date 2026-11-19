"""Единственная проверка на тестовой выборке.

    python -m src.final            # с записью в MLflow
    python -m src.final --no-mlflow

ПРАВИЛО: лучшая модель выбрана до просмотра теста - по средней ошибке
кросс-валидации (cv_mse_mean) и простоте. Ниже она зафиксирована в константе.
Тестовые метрики остальных экспериментов считаются только для итоговой таблицы
отчёта; выбор модели по ним менять нельзя.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV, KFold

from src.data import load_train_test
from src.evaluate import regression_metrics
from src.experiments import EXPERIMENTS, describe_model
from src.split import load_config, load_split
from src.train import MLFLOW_EXPERIMENT, TRACKING_URI, git_info

CHOSEN = "E1c-author-baseline"  # решение принято до просмотра теста
MODEL_PATH = Path("models/model.joblib")


def fit_final(spec, X, y, seed):
    # Обучение на всей обучающей части. Если есть сетка - подбор по кросс-валидации.
    base = spec["build"]()
    grid = spec.get("param_grid")
    if not grid:
        return base.fit(X, y), ""
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(
        base, grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1
    ).fit(X, y)
    return search.best_estimator_, str(search.best_params_)


def save_test_figure(name, y_true, pred, path):
    # Остатки против прогноза и прогноз против истины на тестовой выборке.
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(pred, y_true - pred, s=24)
    ax[0].axhline(0, color="k", lw=1)
    ax[0].set(title=f"{name}: остатки (тест)", xlabel="Прогноз массы, г",
              ylabel="Остаток (факт − прогноз), г")
    top = max(y_true.max(), pred.max())
    ax[1].scatter(y_true, pred, s=24)
    ax[1].plot([0, top], [0, top], "k--", lw=1)
    ax[1].set(title=f"{name}: прогноз и истина (тест)", xlabel="Истинная масса, г",
              ylabel="Прогноз массы, г")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    use_mlflow = not args.no_mlflow

    seed = load_config()["seed"]
    saved = load_split()
    X_train, y_train, X_test, y_test = load_train_test()

    cv_path = Path("runs/cv_results.csv")
    cv_table = pd.read_csv(cv_path, index_col="experiment") if cv_path.exists() else None
    if cv_table is None or set(EXPERIMENTS) - set(cv_table.index):
        print("Внимание: в runs/cv_results.csv есть не все эксперименты. "
              "Сначала выполните: python -m src.train")

    if use_mlflow:
        import mlflow

        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

    Path("runs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    rows, chosen_run_id = [], None
    stamp = time.strftime("%Y%m%d-%H%M%S")

    for name, spec in EXPERIMENTS.items():
        cols = spec["numeric"] + ["Species"]
        model, best_params = fit_final(spec, X_train[cols], y_train, seed)
        pred = model.predict(X_test[cols])
        m = regression_metrics(y_test, pred)
        row = {
            "experiment": name,
            "test_mse": m["mse"],
            "test_mae": m["mae"],
            "test_r2": m["r2"],
            "test_n_nonpositive_pred": int((pred <= 0).sum()),
        }
        if cv_table is not None and name in cv_table.index:
            row.update(cv_table.loc[name].to_dict())
        rows.append(row)
        print(f"{name}: test MSE={m['mse']:.1f} MAE={m['mae']:.1f} R2={m['r2']:.3f}")

        is_chosen = name == CHOSEN
        preds_df = pd.DataFrame({"y_true": y_test, "y_pred": pred}, index=y_test.index)
        preds_df["residual"] = preds_df["y_true"] - preds_df["y_pred"]
        preds_df["abs_error"] = preds_df["residual"].abs()
        preds_df["Species"] = X_test["Species"].values
        preds_path = Path("runs") / f"test_predictions_{name}.csv"
        preds_df.to_csv(preds_path)
        Path("runs/figures").mkdir(parents=True, exist_ok=True)
        fig_path = Path("runs/figures") / f"{name}_test.png"
        save_test_figure(name, y_test, pd.Series(pred, index=y_test.index), fig_path)

        if is_chosen:
            joblib.dump(model, MODEL_PATH)
            details = X_test[cols].copy()
            details["y_true"] = y_test
            details["y_pred"] = pred
            details["residual"] = details["y_true"] - details["y_pred"]
            details["abs_error"] = details["residual"].abs()
            details.to_csv("runs/test_predictions.csv")
            meta = {
                "experiment": name,
                "description": spec["description"],
                "features": cols,
                "target": "Weight (г)",
                "sklearn": sklearn.__version__,
                "python": platform.python_version(),
                "data_sha256": saved["data_sha256"],
                "seed": seed,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "test_metrics": m,
            }
            Path("models/model_meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        if use_mlflow:
            regressor_name, target_transform = describe_model(model)
            with mlflow.start_run(run_name=f"{name}-TEST-{stamp}") as run:
                mlflow.log_params({
                    "experiment": name,
                    "description": spec["description"],
                    "stage": "final test evaluation",
                    "model": regressor_name,
                    "target_transform": target_transform,
                    "features": ",".join(cols),
                    "best_params": best_params or "none",
                    "seed": seed,
                    "split": f"stratified by {saved['stratify_by']}, test_size={saved['test_size']}",
                    "cv": "RepeatedKFold(5 folds x 3 repeats)",
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "data_sha256": saved["data_sha256"],
                    "chosen_as_best": is_chosen,
                })
                metrics = {k: v for k, v in row.items()
                           if k != "experiment" and isinstance(v, (int, float))}
                mlflow.log_metrics(metrics)
                mlflow.set_tags({**git_info(), "python": platform.python_version(),
                                 "sklearn": sklearn.__version__, "pandas": pd.__version__})
                mlflow.log_artifact(str(preds_path), artifact_path="test")
                mlflow.log_artifact(str(fig_path), artifact_path="test")
                if is_chosen:
                    chosen_run_id = run.info.run_id
                    mlflow.log_artifact(str(MODEL_PATH))
                    mlflow.log_artifact("models/model_meta.json")
                    mlflow.log_artifact("runs/test_predictions.csv")
                    try:  # формат MLflow; joblib выше сохранён в любом случае
                        import mlflow.sklearn

                        mlflow.sklearn.log_model(model, artifact_path="model")
                    except Exception as exc:
                        print("   (mlflow.sklearn.log_model не сработал:", exc, ")")

    if chosen_run_id:
        Path("runs/chosen_run_id.txt").write_text(chosen_run_id, encoding="utf-8")

    table = pd.DataFrame(rows).set_index("experiment")
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    table.to_csv("reports/tables/results_table.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n", table[["cv_mse_mean", "cv_mae_mean", "cv_r2_mean",
                       "test_mse", "test_mae", "test_r2"]].round(3).to_string()
          if cv_table is not None else table.round(3).to_string())
    print(f"\nМодель сохранена: {MODEL_PATH}")


if __name__ == "__main__":
    main()
