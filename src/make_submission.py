"""Формирование submission.csv для Kaggle (все 3.37 млн строк)."""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from config import DATA_DIR, RESULTS_DIR, SUBMISSIONS_DIR

FEATURES = [
    "store_nbr", "item_nbr", "family", "city", "state", "type", "cluster", "class", "perishable",
    "onpromotion", "promo_count_store", "promo_ratio_store",
    "oil", "is_holiday", "transactions",
    "dayofweek", "month", "day", "year", "weekofyear", "dayofyear",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_std_7", "roll_mean_30", "roll_std_30", "roll_mean_90", "roll_std_90",
]
CAT_FEATURES = ["store_nbr", "item_nbr", "family", "city", "state", "type", "cluster", "class"]


def build_lags_from_train(features, target_df):
    features = features.sort_values(["store_nbr", "item_nbr", "date"])
    target_pairs = target_df[["store_nbr", "item_nbr"]].drop_duplicates()
    features = features.merge(target_pairs, on=["store_nbr", "item_nbr"], how="inner")

    last_90 = features.groupby(["store_nbr", "item_nbr"], sort=False).tail(90).copy()
    last_90["rank_end"] = last_90.groupby(["store_nbr", "item_nbr"], sort=False).cumcount(ascending=False)

    lag_frames = []
    for lag, rank in [(1, 1), (7, 7), (14, 14), (28, 28)]:
        df_lag = last_90[last_90["rank_end"] == rank][["store_nbr", "item_nbr", "unit_sales"]].rename(
            columns={"unit_sales": f"lag_{lag}"}
        )
        lag_frames.append(df_lag)

    roll = last_90.groupby(["store_nbr", "item_nbr"], sort=False)["unit_sales"].agg(
        roll_mean_7=lambda x: x.tail(7).mean(),
        roll_std_7=lambda x: x.tail(7).std(),
        roll_mean_30=lambda x: x.tail(30).mean(),
        roll_std_30=lambda x: x.tail(30).std(),
        roll_mean_90="mean",
        roll_std_90="std",
    ).reset_index()

    return roll, lag_frames


def main():
    print("=" * 60)
    print("SUBMISSION (все строки)")
    print("=" * 60)

    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    print(f"  test: {test.shape}")

    d = test["date"].dt
    test["dayofweek"] = d.dayofweek.astype("int8")
    test["month"] = d.month.astype("int8")
    test["day"] = d.day.astype("int8")
    test["year"] = d.year.astype("int16")
    test["weekofyear"] = d.isocalendar().week.astype("int8")
    test["dayofyear"] = d.dayofyear.astype("int16")
    test["dow_sin"] = np.sin(2 * np.pi * test["dayofweek"] / 7)
    test["dow_cos"] = np.cos(2 * np.pi * test["dayofweek"] / 7)
    test["month_sin"] = np.sin(2 * np.pi * test["month"] / 12)
    test["month_cos"] = np.cos(2 * np.pi * test["month"] / 12)

    stores_df = pd.read_csv(DATA_DIR / "stores.csv")
    items_df = pd.read_csv(DATA_DIR / "items.csv")
    test = test.merge(stores_df, on="store_nbr", how="left")
    test = test.merge(items_df, on="item_nbr", how="left")

    oil = pd.read_csv(DATA_DIR / "oil.csv", parse_dates=["date"])
    oil = oil.sort_values("date").rename(columns={"dcoilwtico": "oil"})
    oil["oil"] = oil["oil"].ffill().bfill()
    test = test.merge(oil, on="date", how="left")
    test["oil"] = test["oil"].ffill().bfill().fillna(0)

    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])
    national = holidays[holidays["locale"] == "National"]["date"].unique()
    test["is_holiday"] = test["date"].isin(national).astype("int8")

    transactions = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["date"])
    test = test.merge(transactions, on=["date", "store_nbr"], how="left")
    test["transactions"] = test["transactions"].fillna(0).astype("float32")

    features = pd.read_parquet(DATA_DIR / "features.parquet")
    roll, lag_frames = build_lags_from_train(features, test)

    test = test.merge(roll, on=["store_nbr", "item_nbr"], how="left")
    for df_lag in lag_frames:
        test = test.merge(df_lag, on=["store_nbr", "item_nbr"], how="left")

    test["onpromotion"] = test["onpromotion"].fillna(False).astype(int)
    test["promo_count_store"] = test.groupby(["date", "store_nbr"])["onpromotion"].transform("sum")
    test["promo_ratio_store"] = (
        test["promo_count_store"] / test.groupby(["date", "store_nbr"])["onpromotion"].transform("count")
    )

    for col in FEATURES:
        if col in test.columns:
            test[col] = test[col].fillna(0)

    print("Загружаем CatBoost...")
    model = CatBoostRegressor()
    model.load_model(str(RESULTS_DIR / "catboost_model_full.cbm"))

    print("Прогнозируем...")
    X_test = test[FEATURES].copy()
    for col in CAT_FEATURES:
        X_test[col] = X_test[col].astype(str)

    pred_log = model.predict(X_test)
    pred_sales = np.clip(np.expm1(pred_log), 0, 100)
    print(f"Обрезано > 100: {(np.expm1(pred_log) > 100).sum()}")

    submission = pd.DataFrame({"id": test["id"].values, "unit_sales": pred_sales})
    out_path = SUBMISSIONS_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"\nСохранено: {out_path}")
    print(f"Строк:       {len(submission)}")
    print(f"Средние:     {submission['unit_sales'].mean():.3f}")
    print(f"Максимум:    {submission['unit_sales'].max():.1f}")
    print(f"NaN:         {submission['unit_sales'].isna().sum()}")


if __name__ == "__main__":
    main()