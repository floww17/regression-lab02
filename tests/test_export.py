# Проверка приведения таблицы MLflow к читаемому виду (без запуска MLflow).
import pandas as pd

from src.export_mlflow import tidy_runs


def test_tidy_runs_renames_and_filters_columns():
    raw = pd.DataFrame({
        "run_id": ["r1", "r2"],
        "status": ["FINISHED", "FINISHED"],
        "start_time": [2, 1],
        "metrics.cv_mse_mean": [10.0, 20.0],
        "params.model": ["Ridge", "LinearRegression"],
        "tags.mlflow.runName": ["b", "a"],
        "tags.mlflow.user": ["u", "u"],
        "tags.git_commit": ["abc", "abc"],
    })
    out = tidy_runs(raw)
    assert list(out["run_name"]) == ["a", "b"]  # сортировка по времени начала
    assert {"cv_mse_mean", "param_model", "tag_git_commit"} <= set(out.columns)
    assert not any("mlflow.user" in c for c in out.columns)
