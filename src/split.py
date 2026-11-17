"""Разбиение данных на обучающую и тестовую части.

Запуск из корня репозитория:
    python -m src.split

Что делает:
1. Читает параметры из configs/split.json.
2. Один раз делит таблицу на train и test, сохраняя пропорции видов (Species).
3. Записывает номера строк в configs/split_indices.json.
   Тест после этого "заморожен": все остальные скрипты читают этот файл
   и никогда не делят данные заново.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

CONFIG_PATH = Path("configs/split.json")


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def file_sha256(path):
    """Контрольная сумма файла: по ней видно, что данные не менялись."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_split(df, stratify_by, test_size, seed):
    """Возвращает (train_idx, test_idx): списки позиций строк (0..n-1)."""
    positions = list(range(len(df)))
    train_idx, test_idx = train_test_split(
        positions,
        test_size=test_size,
        random_state=seed,
        stratify=df[stratify_by],
    )
    return sorted(train_idx), sorted(test_idx)


def save_split(cfg, train_idx, test_idx):
    payload = {
        "data_path": cfg["data_path"],
        "data_sha256": file_sha256(cfg["data_path"]),
        "stratify_by": cfg["stratify_by"],
        "test_size": cfg["test_size"],
        "seed": cfg["seed"],
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "train_idx": train_idx,
        "test_idx": test_idx,
    }
    out = Path(cfg["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_split(path="configs/split_indices.json"):
    """Читает сохранённые индексы. Используйте это везде вместо нового split."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    cfg = load_config()
    df = pd.read_csv(cfg["data_path"])
    train_idx, test_idx = make_split(
        df, cfg["stratify_by"], cfg["test_size"], cfg["seed"]
    )
    save_split(cfg, train_idx, test_idx)

    print(f"Всего строк: {len(df)}, train: {len(train_idx)}, test: {len(test_idx)}")
    summary = pd.DataFrame(
        {
            "train": df.iloc[train_idx][cfg["stratify_by"]].value_counts(),
            "test": df.iloc[test_idx][cfg["stratify_by"]].value_counts(),
        }
    ).fillna(0).astype(int)
    print(summary)


if __name__ == "__main__":
    main()
