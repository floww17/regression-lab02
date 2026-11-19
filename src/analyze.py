"""Анализ ошибок выбранной модели (графики и таблицы для отчёта).

    python -m src.analyze            # дополнительно кладёт графики в запуск MLflow
    python -m src.analyze --no-mlflow

Нужны файлы после `python -m src.train` и `python -m src.final`.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd  

from src.data import load_train_test 
from src.final import CHOSEN  

FIG = Path("reports/figures")
TAB = Path("reports/tables")


def species_table(df, label):
    g = df.groupby("Species")
    out = pd.DataFrame({
        "n": g.size(),
        "MAE_g": g["abs_error"].mean(),
        "MAE_percent_of_weight": g.apply(lambda d: (d["abs_error"] / d["y_true"]).mean() * 100),
    }).round(1)
    out.insert(0, "sample", label)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    X_train, y_train, _, _ = load_train_test()
    test = pd.read_csv("runs/test_predictions.csv", index_col=0)
    oof = pd.read_csv(f"runs/oof_{CHOSEN}.csv", index_col=0)
    oof["abs_error"] = oof["residual"].abs()
    oof = oof.join(X_train.drop(columns=["Species"]))

    # 1. Остатки и прогноз против истины
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for df, name, color in [(oof, "обучающие (вне фолда)", "tab:blue"), (test, "тест", "tab:red")]:
        ax[0].scatter(df["y_pred"], df["residual"], s=22, alpha=.75, label=name, color=color)
        ax[1].scatter(df["y_true"], df["y_pred"], s=22, alpha=.75, label=name, color=color)
    ax[0].axhline(0, color="k", lw=1)
    ax[0].set(title="Остатки в зависимости от прогноза", xlabel="Прогноз массы, г",
              ylabel="Остаток (факт − прогноз), г")
    top = max(oof["y_true"].max(), test["y_true"].max())
    ax[1].plot([0, top], [0, top], "k--", lw=1)
    ax[1].set(title="Прогноз и истинное значение", xlabel="Истинная масса, г",
              ylabel="Прогноз массы, г")
    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(FIG / "lab02_residuals_and_pred_vs_true.png", dpi=150)
    plt.close(fig)

    # 2. Крупнейшие ошибки с исходными признаками
    cols = ["Species", "Length1", "Length2", "Length3", "Height", "Width",
            "y_true", "y_pred", "residual", "abs_error"]
    oof.sort_values("abs_error", ascending=False).head(5)[cols].round(1).to_csv(
        TAB / "top_errors_oof.csv")
    test_feat = test.copy()
    test_cols = [c for c in cols if c in test_feat.columns]
    test_feat.sort_values("abs_error", ascending=False).head(5)[test_cols].round(1).to_csv(
        TAB / "top_errors_test.csv")

    # 3. MAE по видам и по диапазону размера (границы - по обучающим данным)
    sp = pd.concat([species_table(oof, "обучающие, вне фолда"), species_table(test, "тест")])
    sp.to_csv(TAB / "mae_by_species.csv")

    edges = np.quantile(y_train, [0, 1 / 3, 2 / 3, 1])
    labels = ["малые", "средние", "крупные"]
    rows = []
    for df, label in [(oof, "обучающие, вне фолда"), (test, "тест")]:
        bins = pd.cut(df["y_true"], bins=edges, labels=labels, include_lowest=True)
        for b in labels:
            d = df[bins == b]
            rows.append({"sample": label, "size_group": b, "n": len(d),
                         "MAE_g": round(d["abs_error"].mean(), 1) if len(d) else np.nan,
                         "bias_g (факт-прогноз)": round(d["residual"].mean(), 1) if len(d) else np.nan})
    pd.DataFrame(rows).to_csv(TAB / "mae_by_size.csv", index=False)

    # 4. Отрицательные и физически невозможные прогнозы у всех экспериментов
    neg = []
    for f in sorted(Path("runs").glob("oof_*.csv")):
        d = pd.read_csv(f, index_col=0)
        neg.append({"experiment": f.stem.replace("oof_", ""),
                    "n_pred_le_0": int((d["y_pred"] <= 0).sum()),
                    "min_pred_g": round(d["y_pred"].min(), 1), "n": len(d)})
    pd.DataFrame(neg).to_csv(TAB / "negative_predictions.csv", index=False)

    print("Крупнейшие ошибки (обучающие, вне фолда):")
    print(pd.read_csv(TAB / "top_errors_oof.csv", index_col=0).to_string())
    print("\nMAE по видам:\n", sp.to_string())
    print("\nMAE по размеру:\n", pd.read_csv(TAB / "mae_by_size.csv").to_string(index=False))
    print("\nПрогнозы <= 0:\n", pd.read_csv(TAB / "negative_predictions.csv").to_string(index=False))

    run_file = Path("runs/chosen_run_id.txt")
    if not args.no_mlflow and run_file.exists():
        import mlflow

        from src.train import TRACKING_URI
        mlflow.set_tracking_uri(TRACKING_URI)
        with mlflow.start_run(run_id=run_file.read_text().strip()):
            mlflow.log_artifact(str(FIG / "lab02_residuals_and_pred_vs_true.png"))
            for f in TAB.glob("*.csv"):
                mlflow.log_artifact(str(f), artifact_path="tables")


if __name__ == "__main__":
    main()
