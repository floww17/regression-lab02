"""Описание экспериментов. Каждая модель - единая цепочка (Pipeline):
заполнение пропусков -> (логарифм) -> масштаб/кодирование -> модель."""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data import CATEGORICAL, NUMERIC_ALL
from src.split import load_config

SEED = load_config()["seed"]


def make_preprocessor(numeric, categorical, log_features=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if log_features:
        num_steps.append(
            ("log", FunctionTransformer(np.log, feature_names_out="one-to-one"))
        )
    num_steps.append(("scale", StandardScaler()))
    cat = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("num", Pipeline(num_steps), numeric), ("cat", cat, categorical)]
    )


def make_model(regressor, numeric, categorical, log_features=False, target="raw"):
    pipe = Pipeline(
        [
            ("preprocess", make_preprocessor(numeric, categorical, log_features)),
            ("regressor", regressor),
        ]
    )
    if target == "log1p":
        return TransformedTargetRegressor(
            regressor=pipe, func=np.log1p, inverse_func=np.expm1
        )
    if target == "log":
        return TransformedTargetRegressor(
            regressor=pipe, func=np.log, inverse_func=np.exp
        )
    return pipe


def describe_model(model):
    """Класс итогового регрессора и преобразование цели (для журнала MLflow)."""
    target = "none"
    m = model
    if isinstance(m, TransformedTargetRegressor):
        target = getattr(m.func, "__name__", "custom")
        m = m.regressor
    if isinstance(m, Pipeline):
        m = m.steps[-1][1]
    return type(m).__name__, target


AUTHOR_NUMERIC = ["Length3", "Height", "Width"]  # так делал автор проекта

# Сетки гиперпараметров (заданы заранее, чтобы не подгонять под результат)
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
_FOREST = {"max_depth": [4, 8, None], "min_samples_leaf": [1, 2, 4]}
FOREST_GRID_RAW = {f"regressor__{k}": v for k, v in _FOREST.items()}
FOREST_GRID_LOG = {f"regressor__regressor__{k}": v for k, v in _FOREST.items()}

EXPERIMENTS = {
    "E0-mean": dict(
        description="Наивный прогноз: среднее обучающей выборки",
        numeric=NUMERIC_ALL,
        build=lambda: make_model(DummyRegressor(strategy="mean"), NUMERIC_ALL, CATEGORICAL),
    ),
    "E0-median": dict(
        description="Наивный прогноз: медиана обучающей выборки",
        numeric=NUMERIC_ALL,
        build=lambda: make_model(DummyRegressor(strategy="median"), NUMERIC_ALL, CATEGORICAL),
    ),
    "E1a-linear-raw": dict(
        description="Линейная регрессия, все длины, масса без преобразования",
        numeric=NUMERIC_ALL,
        build=lambda: make_model(LinearRegression(), NUMERIC_ALL, CATEGORICAL),
    ),
    "E1b-linear-log1p-target": dict(
        description="Линейная регрессия, все длины, цель log(1+y)",
        numeric=NUMERIC_ALL,
        build=lambda: make_model(
            LinearRegression(), NUMERIC_ALL, CATEGORICAL, target="log1p"
        ),
    ),
    "E1c-author-baseline": dict(
        description="Базовое решение автора: Length3, Height, Width, лог признаков и цели",
        numeric=AUTHOR_NUMERIC,
        build=lambda: make_model(
            LinearRegression(), AUTHOR_NUMERIC, CATEGORICAL, log_features=True, target="log"
        ),
    ),
    # ---- E2: гребневая регрессия, alpha выбирается кросс-валидацией ----
    "E2a-ridge-author-features": dict(
        description="Ridge, признаки автора (Length3, Height, Width), лог признаков и цели",
        numeric=AUTHOR_NUMERIC,
        build=lambda: make_model(
            Ridge(), AUTHOR_NUMERIC, CATEGORICAL, log_features=True, target="log"
        ),
        param_grid={"regressor__regressor__alpha": ALPHAS},
    ),
    "E2b-ridge-all-lengths": dict(
        description="Ridge, все три длины (сильно коррелируют), лог признаков и цели",
        numeric=NUMERIC_ALL,
        build=lambda: make_model(
            Ridge(), NUMERIC_ALL, CATEGORICAL, log_features=True, target="log"
        ),
        param_grid={"regressor__regressor__alpha": ALPHAS},
    ),
    # ---- E3: случайный лес, небольшая заранее заданная сетка ----
    "E3a-forest-raw-target": dict(
        description="Случайный лес, масса без преобразования",
        numeric=NUMERIC_ALL,
        build=lambda: make_model(
            RandomForestRegressor(n_estimators=100, random_state=SEED),
            NUMERIC_ALL, CATEGORICAL,
        ),
        param_grid=FOREST_GRID_RAW,
    ),
    "E3b-forest-log-target": dict(
        description="Случайный лес, цель log(y)",
        numeric=NUMERIC_ALL,
        build=lambda: make_model(
            RandomForestRegressor(n_estimators=100, random_state=SEED),
            NUMERIC_ALL, CATEGORICAL, target="log",
        ),
        param_grid=FOREST_GRID_LOG,
    ),
    # ---- E4: влияет ли признак объёма V = Length3 * Height * Width ----
    # Одна и та же модель (Ridge, alpha по кросс-валидации), те же данные и разбиение;
    # отличается только наличие V_proxy.
    "E4a-ridge-raw-without-V": dict(
        description="Ridge без логарифмов, без V_proxy (Length3, Height, Width)",
        numeric=AUTHOR_NUMERIC,
        build=lambda: make_model(Ridge(), AUTHOR_NUMERIC, CATEGORICAL),
        param_grid={"regressor__alpha": ALPHAS},
    ),
    "E4b-ridge-raw-with-V": dict(
        description="Ridge без логарифмов, с V_proxy",
        numeric=AUTHOR_NUMERIC + ["V_proxy"],
        build=lambda: make_model(Ridge(), AUTHOR_NUMERIC + ["V_proxy"], CATEGORICAL),
        param_grid={"regressor__alpha": ALPHAS},
    ),
    # Сравнивать с E2a: та же модель в логарифмах, но с V_proxy
    "E4c-ridge-log-with-V": dict(
        description="Ridge в логарифмах (признаки и цель) с V_proxy; сравнить с E2a",
        numeric=AUTHOR_NUMERIC + ["V_proxy"],
        build=lambda: make_model(
            Ridge(), AUTHOR_NUMERIC + ["V_proxy"], CATEGORICAL,
            log_features=True, target="log",
        ),
        param_grid={"regressor__regressor__alpha": ALPHAS},
    ),
}
