"""Экспорт журнала MLflow в файлы репозитория.

    python -m src.export_mlflow
Результат:
    reports/tables/mlflow_runs.csv        - все запуски: параметры, метрики, теги
    reports/tables/mlflow_structure.txt   - снимок структуры запусков и артефактов
"""
import os
from pathlib import Path

import pandas as pd

from src.train import MLFLOW_EXPERIMENT, TRACKING_URI

TAB = Path("reports/tables")


def tidy_runs(df):
    """Приводит таблицу mlflow.search_runs к читаемому виду."""
    out = pd.DataFrame(index=df.index)
    out["run_name"] = df.get("tags.mlflow.runName")
    for col in ("run_id", "status", "start_time", "end_time"):
        if col in df.columns:
            out[col] = df[col]
    for col in df.columns:
        if col.startswith("metrics."):
            out[col[len("metrics."):]] = df[col]
        elif col.startswith("params."):
            out["param_" + col[len("params."):]] = df[col]
        elif col.startswith("tags.") and not col.startswith("tags.mlflow."):
            out["tag_" + col[len("tags."):]] = df[col]
    sort_col = "start_time" if "start_time" in out.columns else "run_name"
    return out.sort_values(sort_col).reset_index(drop=True)


def structure_snapshot(root="mlruns"):
    """Дерево каталогов артефактов (имена файлов), если папка mlruns существует."""
    root = Path(root)
    if not root.exists():
        return "(папка mlruns не найдена)\n"
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        depth = Path(dirpath).relative_to(root).parts
        lines.append("  " * len(depth) + (depth[-1] if depth else str(root)) + "/")
        for f in sorted(filenames):
            lines.append("  " * (len(depth) + 1) + f)
    return "\n".join(lines) + "\n"


def main():
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT])
    if runs.empty:
        raise SystemExit("В журнале нет запусков. Сначала: python -m src.train")
    table = tidy_runs(runs)
    TAB.mkdir(parents=True, exist_ok=True)
    table.to_csv(TAB / "mlflow_runs.csv", index=False)
    (TAB / "mlflow_structure.txt").write_text(structure_snapshot(), encoding="utf-8")
    print(f"Запусков: {len(table)}")
    print(table[["run_name", "status"]].to_string(index=False))
    print("\nСохранено: reports/tables/mlflow_runs.csv, mlflow_structure.txt")


if __name__ == "__main__":
    main()
