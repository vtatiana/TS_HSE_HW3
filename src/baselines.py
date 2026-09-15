"""Бейзлайн-модели."""
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
from config import DATA_DIR, RESULTS_DIR

TRAIN_END = "2016-12-31"
VAL_START = "2017-01-01"
VAL_END = "2017-08-15"


def rmsle(y_true, y_pred):
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def naive(y_train, n):
    return np.full(n, y_train[-1])


def seasonal_naive(y_train, n, season=7):
    if len(y_train) < season:
        return np.full(n, y_train[-1])
    return np.array([y_train[-season + i % season] for i in range(n)])


def auto_ets(y_train, n):
    try:
        model = ExponentialSmoothing(y_train, seasonal_periods=7, trend="add", seasonal="add")
        fit = model.fit(optimized=True)
        return np.clip(np.asarray(fit.forecast(n)), 0, None)
    except Exception as e:
        print(f"    ETS failed: {e}, fallback to naive")
        return naive(y_train, n)


def auto_theta(y_train, n):
    try:
        model = ThetaModel(y_train, period=7)
        fit = model.fit()
        return np.clip(np.asarray(fit.forecast(n)), 0, None)
    except Exception as e:
        print(f"    Theta failed: {e}, fallback to naive")
        return naive(y_train, n)


def seasonal_naive_per_item(train_part, val_part, season=7):
    print("\nSeasonalNaive на уровне store-item...")
    last_week = (train_part.sort_values("date")
                 .groupby(["store_nbr", "item_nbr"]).tail(season).copy())
    last_week["day_rank"] = last_week.groupby(["store_nbr", "item_nbr"]).cumcount()

    lookup = last_week.set_index(["store_nbr", "item_nbr", "day_rank"])["unit_sales"]

    val = val_part[["date", "store_nbr", "item_nbr", "unit_sales"]].copy()
    val["day_rank"] = (val["date"] - val["date"].min()).dt.days % season
    val["key"] = list(zip(val["store_nbr"], val["item_nbr"], val["day_rank"]))
    lookup_dict = lookup.to_dict()
    val["pred"] = val["key"].map(lookup_dict).fillna(0).clip(lower=0)

    y_true = np.log1p(val["unit_sales"].clip(lower=0).values)
    y_pred = np.log1p(val["pred"].values)
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def main():
    print("Загружаем данные...")
    train = pd.read_parquet(DATA_DIR / "train_subsample.parquet")

    train_part = train[train["date"] <= TRAIN_END]
    val_part = train[(train["date"] >= VAL_START) & (train["date"] <= VAL_END)]
    print(f"Train: {train_part.shape}, Val: {val_part.shape}")

    y_train = train_part.groupby("date")["unit_sales"].sum().sort_index().values
    y_val = val_part.groupby("date")["unit_sales"].sum().sort_index().values
    print(f"Train дней: {len(y_train)}, Val дней: {len(y_val)}")

    n = len(y_val)
    results = {}

    print("\nОбучаем бейзлайны...")
    for name, fn in [("Naive", naive), ("SeasonalNaive", seasonal_naive),
                     ("auto_ets", auto_ets), ("auto_theta", auto_theta)]:
        print(f"  {name}...")
        pred = fn(y_train, n)
        score = rmsle(y_val, pred)
        results[name] = score
        print(f"    RMSLE = {score:.4f}")

    score_sn_item = seasonal_naive_per_item(train_part, val_part)
    print(f"  SeasonalNaive (store-item): RMSLE = {score_sn_item:.4f}")
    results["SeasonalNaive_itemlevel"] = score_sn_item

    print("\n=== ИТОГ ===")
    df = pd.DataFrame(list(results.items()), columns=["model", "RMSLE"]).sort_values("RMSLE")
    print(df.to_string(index=False))
    df.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)
    print(f"\nСохранено: {RESULTS_DIR / 'baseline_results.csv'}")


if __name__ == "__main__":
    main()