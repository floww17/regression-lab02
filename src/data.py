"""Загрузка данных по сохранённому разбиению (configs/split_indices.json)."""
import pandas as pd

from src.split import file_sha256, load_config, load_split

NUMERIC_ALL = ["Length1", "Length2", "Length3", "Height", "Width"]
CATEGORICAL = ["Species"]
TARGET = "Weight"


def load_train_test():
    """Возвращает X_train, y_train, X_test, y_test.

    Индекс строк (номера в исходном CSV) сохраняется: по нему потом
    находим конкретные ошибки модели.
    """
    cfg = load_config()
    saved = load_split()
    if file_sha256(cfg["data_path"]) != saved["data_sha256"]:
        raise ValueError(
            "CSV изменился после разбиения. Пересоздайте разбиение: python -m src.split"
        )
    df = pd.read_csv(cfg["data_path"])
    train = df.iloc[saved["train_idx"]]
    test = df.iloc[saved["test_idx"]]
    # Производный признак: приближённый объём. Считается только из входных
    # измерений строки, масса (цель) в нём не используется.
    df = df.assign(V_proxy=df["Length3"] * df["Height"] * df["Width"])
    train = df.iloc[saved["train_idx"]]
    test = df.iloc[saved["test_idx"]]
    feats = NUMERIC_ALL + ["V_proxy"] + CATEGORICAL
    return train[feats], train[TARGET], test[feats], test[TARGET]
