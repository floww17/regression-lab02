"""Мультиколлинеарность: корреляции, VIF и устойчивость коэффициентов.

    python -m src.collinearity
Все расчёты - только по обучающей части. Результаты: reports/tables/*.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from src.data import load_train_test
from src.experiments import make_preprocessor
from src.split import load_config

TAB = Path("reports/tables")
RIDGE_ALPHA = 1.0  # фиксированное значение для наглядного сравнения устойчивости
N_BOOT = 200


def vif(df):
    """VIF_j = 1 / (1 - R^2_j): R^2 регрессии столбца j на остальные столбцы."""
    out = {}
    for col in df.columns:
        y = df[col].to_numpy(float)
        X = np.column_stack([np.ones(len(df)), df.drop(columns=col).to_numpy(float)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        r2 = 1 - resid.var() / y.var()
        out[col] = np.inf if r2 > 1 - 1e-9 else 1 / (1 - r2)
    return pd.Series(out)


def bootstrap_coefs(X, y, numeric, log_features, log_target, regressor, seed):
    """Коэффициенты при числовых признаках (стандартизованных) по бутстреп-выборкам."""
    rng = np.random.default_rng(seed)
    cols = numeric + ["Species"]
    coefs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(X), len(X))
        Xb, yb = X.iloc[idx][cols], y.iloc[idx]
        Z = make_preprocessor(numeric, ["Species"], log_features).fit_transform(Xb)
        if hasattr(Z, "toarray"):
            Z = Z.toarray()
        target = np.log(yb) if log_target else yb
        coefs.append(regressor().fit(Z, target).coef_[: len(numeric)])
    return pd.DataFrame(coefs, columns=numeric)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    seed = load_config()["seed"]
    X, y, _, _ = load_train_test()
    raw_cols = ["Length1", "Length2", "Length3", "Height", "Width", "V_proxy"]

    corr = X[raw_cols].corr().round(3)
    corr_log = np.log(X[raw_cols]).corr().round(3)
    corr.to_csv(TAB / "correlations_raw.csv")
    corr_log.to_csv(TAB / "correlations_log.csv")
    print("Корреляции (исходная шкала):\n", corr.to_string())
    print("\nКорреляции (логарифмы):\n", corr_log.to_string())

    sets = {
        "L3, H, W": ["Length3", "Height", "Width"],
        "L3, H, W, V": ["Length3", "Height", "Width", "V_proxy"],
        "L1, L2, L3, H, W": ["Length1", "Length2", "Length3", "Height", "Width"],
    }
    rows = []
    for label, cols in sets.items():
        for scale, data in [("исходная", X[cols]), ("логарифм", np.log(X[cols]))]:
            for feat, v in vif(data).items():
                rows.append({"набор": label, "шкала": scale, "признак": feat,
                             "VIF": "inf" if np.isinf(v) else round(v, 1)})
    vif_table = pd.DataFrame(rows)
    vif_table.to_csv(TAB / "vif.csv", index=False)
    print("\nVIF:\n", vif_table.to_string(index=False))

    scenarios = [
        ("E4a: L3,H,W, сырые", ["Length3", "Height", "Width"], False, False),
        ("E4b: L3,H,W,V, сырые", ["Length3", "Height", "Width", "V_proxy"], False, False),
        ("E1c: L3,H,W, логарифмы", ["Length3", "Height", "Width"], True, True),
        ("E4c: L3,H,W,V, логарифмы", ["Length3", "Height", "Width", "V_proxy"], True, True),
        ("E2b: L1,L2,L3,H,W, логарифмы",
         ["Length1", "Length2", "Length3", "Height", "Width"], True, True),
    ]
    rows = []
    for label, numeric, log_f, log_t in scenarios:
        for name, reg in [("МНК", LinearRegression), (f"Ridge a={RIDGE_ALPHA}",
                                                      lambda: Ridge(alpha=RIDGE_ALPHA))]:
            b = bootstrap_coefs(X, y, numeric, log_f, log_t, reg, seed)
            med_sign = np.sign(b.median())
            for feat in numeric:
                rows.append({
                    "сценарий": label, "модель": name, "признак": feat,
                    "коэф., среднее": round(b[feat].mean(), 3),
                    "коэф., std": round(b[feat].std(), 3),
                    "доля смены знака": round(float((np.sign(b[feat]) != med_sign[feat]).mean()), 3),
                })
    stab = pd.DataFrame(rows)
    stab.to_csv(TAB / "coef_stability.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\nУстойчивость коэффициентов (бутстреп, {N_BOOT} выборок):\n", stab.to_string(index=False))


if __name__ == "__main__":
    main()
