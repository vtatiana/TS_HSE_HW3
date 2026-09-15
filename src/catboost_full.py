"""Полная версия CatBoost (10 млн строк, depth=8, 1500 итераций)."""
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from config import DATA_DIR, RESULTS_DIR

TRAIN_END = "2016-12-31"
VAL_START = "2017-01-01"
VAL_END = "2017-08-15"

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


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def main():
    print("=" * 60)
    print("CATBOOST FULL")
    print("=" * 60)

    df = pd.read_parquet(DATA_DIR / "features.parquet")
    print(f"Загружено: {df.shape}")

    train = df[df["date"] <= TRAIN_END]
    val = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)]

    N_TRAIN = 10_000_000
    if len(train) > N_TRAIN:
        train = train.sample(n=N_TRAIN, random_state=42).reset_index(drop=True)

    print(f"  Train (subsample): {train.shape}")
    print(f"  Val (полный):      {val.shape}")

    X_train = train[FEATURES].copy()
    y_train = train["target_log"].values
    X_val = val[FEATURES].copy()
    y_val = val["target_log"].values

    for col in CAT_FEATURES:
        X_train[col] = X_train[col].astype(str)
        X_val[col] = X_val[col].astype(str)

    train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
    val_pool = Pool(X_val, y_val, cat_features=CAT_FEATURES)

    print("\nОбучаем CatBoost...")
    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        od_type="Iter",
        od_wait=50,
        verbose=50,
        task_type="CPU",
        thread_count=-1,
        subsample=0.8,
        bootstrap_type="Bernoulli",
        save_snapshot=True,
        snapshot_file=str(RESULTS_DIR / "catboost_full_snapshot.cbs"),
        snapshot_interval=100,
        allow_writing_files=True,
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    print("\nСчитаем метрики...")
    pred_log = model.predict(val_pool)
    rmsle_val = rmsle(y_val, pred_log)
    print(f"RMSLE (val, log-пространство): {rmsle_val:.4f}")

    model_path = RESULTS_DIR / "catboost_model_full.cbm"
    model.save_model(str(model_path))
    print(f"Модель: {model_path}")

    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.get_feature_importance(train_pool),
    }).sort_values("importance", ascending=False)
    imp.to_csv(RESULTS_DIR / "catboost_importance_full.csv", index=False)
    print("\nTop-15 фичей:")
    print(imp.head(15).to_string(index=False))

    pd.DataFrame([{"model": "CatBoost_full", "RMSLE": rmsle_val}]).to_csv(
        RESULTS_DIR / "catboost_results_full.csv", index=False
    )
    print(f"\nСохранено: {RESULTS_DIR / 'catboost_results_full.csv'}")


if __name__ == "__main__":
    main()