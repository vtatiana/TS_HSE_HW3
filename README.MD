# TS HSE 2025-2026 HW3: Favorita Sales Forecasting

## Описание
Прогнозирование ежедневных продаж сети магазинов Corporación Favorita
(Эквадор) с использованием экзогенных признаков.

## Данные
Скачать с Kaggle: https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting

Положить в папку `data/`:
- `train.csv`, `test.csv`, `stores.csv`, `items.csv`
- `oil.csv`, `holidays_events.csv`, `transactions.csv`
- `sample_submission.csv`

## Установка
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## Запуск

Полный пайплайн:
python src/run_experiment.py

Только подготовка данных:
python src/run_experiment.py --only-prepare

С LSTM:
python src/run_experiment.py --with-lstm

## Структура
- `src/config.py` — параметры проекта
- `src/data_loader.py` — загрузка train
- `src/prepare_data.py` — подготовка подвыборки
- `src/features.py` — генерация признаков
- `src/eda.py` — разведочный анализ
- `src/baselines.py` — бейзлайны
- `src/catboost_full.py` — CatBoost (ML)
- `src/lstm_model.py` — LSTM (DL)
- `src/error_analysis.py` — анализ ошибок
- `src/make_submission.py` — формирование submission
- `src/run_experiment.py` — единая точка входа

## Метрика
RMSLE (Root Mean Squared Logarithmic Error) — как в соревновании Kaggle.

## Результаты
- CatBoost: RMSLE = **0.4677** (store-item)
- LSTM: RMSLE = **0.6054** (store-item)
- Kaggle Public Score: **1.03390**

См. `results/report.md` — полный отчёт.