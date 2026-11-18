"""Паспорт данных: фактические диапазоны и проверки считаются по файлу.

    python -m src.passport
Результат: reports/tables/data_passport.csv и data_passport.md
"""
from pathlib import Path

import pandas as pd

from src.split import load_config

# Что нельзя вычислить по данным, описано здесь (и отделено от проверенных фактов).
STATIC = {
    "Species": ("вид рыбы", "категориальный", "словарь категорий",
                "признак; ключ стратификации разбиения (группой не является)"),
    "Weight": ("масса рыбы", "числовой", "граммы",
               "целевая переменная"),
    "Length1": ("первая длина", "числовой", "см",
                "признак (сильно коррелирует с Length2, Length3)"),
    "Length2": ("вторая длина", "числовой", "см",
                "признак (сильно коррелирует с Length1, Length3)"),
    "Length3": ("третья длина (предположительно наибольшая)", "числовой", "см",
                "признак"),
    "Height": ("высота тела", "числовой", "см", "признак"),
    "Width": ("ширина тела", "числовой", "см", "признак"),
    "V_proxy": ("приближённый объём = Length3·Height·Width", "числовой, производный",
                "см³ (условно)", "признак эксперимента E4; цель не используется"),
}


def _range(s):
    return f"{s.min():.2f} – {s.max():.2f}"


def build_passport(df):
    df = df.assign(V_proxy=df["Length3"] * df["Height"] * df["Width"])
    rows = []
    for name, (meaning, dtype, units, role) in STATIC.items():
        s = df[name]
        missing = int(s.isna().sum())
        if name == "Species":
            counts = s.value_counts()
            clean = s.str.strip().str.lower()
            checks = (f"{s.nunique()} видов; пустых строк: {int((s.str.strip() == '').sum())}; "
                      f"дубликатов написания: {s.nunique() - clean.nunique()}; "
                      f"самый редкий: {counts.idxmin()} ({counts.min()})")
            values = ", ".join(f"{k} ({v})" for k, v in counts.items())
        else:
            values = _range(s)
            checks = f"значений ≤ 0: {int((s <= 0).sum())}"
            if name == "Length2":
                checks += f"; Length1 > Length2: {int((df['Length1'] > df['Length2']).sum())}"
            if name == "Length3":
                checks += (f"; Length2 > Length3: {int((df['Length2'] > df['Length3']).sum())}")
        rows.append({
            "Поле": name, "Смысл": meaning, "Тип": dtype, "Единицы": units,
            "Роль": role, "Пропуски": f"{missing} из {len(df)}",
            "Диапазон / словарь": values, "Проверки": checks,
        })
    return pd.DataFrame(rows)


def to_markdown(df):
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def main():
    cfg = load_config()
    df = pd.read_csv(cfg["data_path"])
    passport = build_passport(df)
    out = Path("reports/tables")
    out.mkdir(parents=True, exist_ok=True)
    passport.to_csv(out / "data_passport.csv", index=False)
    (out / "data_passport.md").write_text(to_markdown(passport) + "\n", encoding="utf-8")
    print(to_markdown(passport))
    print(f"\nСтрок: {len(df)}, полных дубликатов: {int(df.duplicated().sum())}")


if __name__ == "__main__":
    main()
