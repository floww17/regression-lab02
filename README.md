# FishGrow: прогноз массы рыбы (лабораторная работа №2)

Воспроизводимая регрессионная модель массы рыбы с честным разбиением данных, сравнением моделей и журналом экспериментов в MLflow. Работа выполнена на основе учебного проекта `fish_weight_prediction_machine_learning`; его исходные файлы (`train.py`, `preprocess.py`, `predict.py`, `reg.joblib`, `Pipfile`) оставлены в корне как справочный baseline и **не используются** в новом конвейере.

- Ветка работы: `lab/02-regression`
- Отчёт: [`reports/lab02-report.md`](reports/lab02-report.md)
- Паспорт модели: [`models/model-card.md`](models/model-card.md)

## Задача

По длине, высоте, ширине и виду предсказать массу рыбы в граммах. Метрики: MSE, MAE, R².

## Результат

Лучшая модель — E1c (линейная регрессия по логарифмам признаков и цели). Тест, 23 строки: MSE 1 574, MAE 25.0 г, R² 0.985. Наивный прогноз средним: MAE 268.5 г. Полная таблица экспериментов E0–E4 — в отчёте.

## Структура репозитория

```
configs/          split.json (параметры разбиения), split_indices.json (индексы train/test)
data/             fish_participant.csv
models/           model.joblib, model_meta.json, model-card.md
reports/          lab02-report.md, figures/, tables/
runs/             временные результаты запусков (не коммитятся)
src/              split.py, data.py, evaluate.py, experiments.py, train.py, final.py,
                  analyze.py, passport.py, collinearity.py, export_mlflow.py
tests/            автоматические проверки
requirements-course.txt, requirements-lock.txt
```

## Быстрый старт (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-course.txt
python -m src.split        # создаёт configs/split_indices.json (однократно)
python -m pytest -q        # ожидается: 23 passed
python -m src.train        # кросс-валидация всех экспериментов, запись в MLflow
python -m src.final        # единственная проверка на тесте, сохранение модели
python -m src.analyze      # графики и таблицы анализа ошибок
python -m src.passport     # паспорт данных
python -m src.collinearity # корреляции, VIF, устойчивость коэффициентов
python -m src.export_mlflow  # экспорт журнала MLflow в reports/tables
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Запуск части экспериментов: `python -m src.train E1c-author-baseline E2a-ridge-author-features`. Запуск без MLflow: `--no-mlflow`.

Использование сохранённой модели:

```python
import joblib, pandas as pd
model = joblib.load("models/model.joblib")
print(model.predict(pd.DataFrame([{"Species": "Perch", "Length3": 30.0, "Height": 8.0, "Width": 4.5}])))
```

## Воспроизводимость

| Параметр                               | Значение                                                                       |
| -------------------------------------- | ------------------------------------------------------------------------------ |
| Python                                 | `1.12.9`                                                                       |
| scikit-learn / pandas / numpy / mlflow | `1.9.1 / 3.1.1 / 2.5.5 / 3.18.0`                                               |
| ОС                                     | `Windows`                                                                      |
| Seed                                   | 42 (разбиение, кросс-валидация, случайный лес)                                 |
| Данные                                 | `data/fish_participant.csv`, SHA-256 в `configs/split_indices.json`            |
| Разбиение                              | 80/20, стратификация по `Species`: 88 строк обучения, 23 теста                 |
| Кросс-валидация                        | `RepeatedKFold` 5×3 на обучающей части; подбор параметров — вложенная проверка |

Метрики воспроизводятся при тех же версиях библиотек; при других версиях возможны небольшие расхождения.

## Данные

`data/fish_participant.csv`: 111 строк, 7 видов, пропусков нет. Поля: `Species`, `Weight` (г), `Length1`, `Length2`, `Length3`, `Height`, `Width` (см).

Файл `fish_holdout_demo.csv` целиком входит в `fish_participant.csv`, поэтому как тестовая выборка не используется. Тест выделен из `fish_participant.csv` и открывается один раз (`src/final.py`).

## Ограничения

Малая выборка (23 тестовые строки), нет сведений о фермах и партиях, возможна ошибка в одной из строк, неизвестные виды и размеры вне обучающего диапазона не поддерживаются. Подробнее — в отчёте и паспорте модели.

## Использование ИИ

Для генерации и отладки кода применялся Claude (Anthropic); подробности и проверка результата — в разделе 9 отчёта и в описании запроса на слияние.
