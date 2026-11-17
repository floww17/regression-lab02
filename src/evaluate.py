"""Метрики и кросс-валидация."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict, cross_validate

SCORING = {
    "mse": "neg_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def regression_metrics(y_true, y_pred):
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_cv(seed, n_splits=5, n_repeats=3):
    return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)


def cv_metrics(model, X, y, seed):
    """Среднее и стандартное отклонение метрик по всем фолдам."""
    res = cross_validate(model, X, y, cv=make_cv(seed), scoring=SCORING)
    out = {}
    for name in SCORING:
        values = res[f"test_{name}"]
        if name in ("mse", "mae"):  # sklearn возвращает отрицательные значения
            values = -values
        out[f"cv_{name}_mean"] = float(np.mean(values))
        out[f"cv_{name}_std"] = float(np.std(values))
    return out


def oof_predictions(model, X, y, seed, n_splits=5):
    """Прогноз для каждой обучающей строки моделью, которая эту строку не видела."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, X, y, cv=cv)
    out = pd.DataFrame({"y_true": y.values, "y_pred": pred}, index=y.index)
    out["residual"] = out["y_true"] - out["y_pred"]
    out["Species"] = X["Species"].values
    return out
