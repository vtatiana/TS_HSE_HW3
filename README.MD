# TS HSE HW3 — Favorita Sales Forecasting



Исходная задача проекта: - Прогноз дневных продаж для сети магазинов Corporación Favorita (Эквадор).
Кратко, что получилось сделать: catboost дал rmsle **0.4677** на валидации и **1.0339** на kaggleборде.

## Что внутри проекта

- `src/` — весь код
- `results/` — отчёт, графики, ноутбук
- `submissions/` — submission для kaggle
- `data/` — сюда кладём исходные csv 

## Данные с kaggle

Скачать с kaggle: https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting

Положили файлики в `data/`:
- `train.csv`,`test.csv`,`stores.csv`,`items.csv`
- `oil.csv`,`holidays_events.csv`,`transactions.csv`
- `sample_submission.csv`

## Установка venv внутри папки

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск скриптов

Полный пайплайн:
```
python src/run_experiment.py
```

Только этап с подготовкой данных:
```
python src/run_experiment.py --only-prepare
```

С моделькой catboost и lstm:
```
python src/run_experiment.py --with-lstm
```

## Результаты

| Модель | RMSLE (store-item) |
|---|---|
| SeasonalNaive | 0.7883 |
| LSTM | 0.6054 |
| **CatBoost** | **0.4677** |
| Kaggle Public | 1.03390 |

Полный отчёт можно посмотреть в `results/report.md`.

## Все файлы проекта

- `src/config.py` — параметры
- `src/data_loader.py` — загрузка train
- `src/prepare_data.py` — подвыборка для модели
- `src/features.py` — фичи
- `src/eda.py` — разведочный анализ
- `src/baselines.py` — бейзлайны
- `src/catboost_full.py` — CatBoost (MLмодель)
- `src/lstm_model.py` — LSTM (DLмодель)
- `src/error_analysis.py` — анализ ошибок
- `src/make_submission.py` — сам submission
- `src/run_experiment.py` — точка входа

## главная метрика

RMSLE — как в соревновании kaggle.