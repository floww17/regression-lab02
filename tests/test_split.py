# Быстрые проверки разбиения. Запуск из корня: python -m pytest -q
import pandas as pd

from src.split import load_config, load_split, make_split


def _data():
    cfg = load_config()
    return cfg, pd.read_csv(cfg["data_path"])


def test_train_and_test_do_not_overlap():
    saved = load_split()
    assert set(saved["train_idx"]).isdisjoint(saved["test_idx"])


def test_split_covers_all_rows_exactly_once():
    cfg, df = _data()
    saved = load_split()
    all_idx = sorted(saved["train_idx"] + saved["test_idx"])
    assert all_idx == list(range(len(df)))


def test_every_species_present_in_train():
    cfg, df = _data()
    saved = load_split()
    species_all = set(df[cfg["stratify_by"]])
    species_train = set(df.iloc[saved["train_idx"]][cfg["stratify_by"]])
    assert species_all == species_train


def test_split_is_deterministic_and_matches_saved_file():
    cfg, df = _data()
    saved = load_split()
    train_idx, test_idx = make_split(
        df, cfg["stratify_by"], cfg["test_size"], cfg["seed"]
    )
    assert train_idx == saved["train_idx"]
    assert test_idx == saved["test_idx"]