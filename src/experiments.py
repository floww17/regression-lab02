"""Описание экспериментов. Каждая модель - единая цепочка (Pipeline):
заполнение пропусков -> (логарифм) -> масштаб/кодирование -> модель."""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data import CATEGORICAL, NUMERIC_ALL


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


AUTHOR_NUMERIC = ["Length3", "Height", "Width"]  # так делал автор проекта

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
}
