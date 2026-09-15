"""Анализ ошибок CatBoost на валидации."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
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

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)


def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def main():
    print("Загружаем features.parquet...")
    df = pd.read_parquet(DATA_DIR / "features.parquet")
    val = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)].copy()
    print(f"Val: {val.shape}")

    X_val = val[FEATURES].copy()
    y_val = val["target_log"].values
    for col in CAT_FEATURES:
        X_val[col] = X_val[col].astype(str)

    print("Загружаем модель...")
    model = CatBoostRegressor()
    model.load_model(str(RESULTS_DIR / "catboost_model_full.cbm"))

    print("Считаем прогнозы...")
    pred = model.predict(X_val)
    val["pred_log"] = pred
    val["error"] = val["pred_log"] - val["target_log"]
    val["sq_error"] = val["error"] ** 2

    print(f"\nОбщий RMSLE: {rmsle(y_val, pred):.4f}")

    # 1. По дням недели
    fig, ax = plt.subplots()
    dow = val.groupby("dayofweek")["sq_error"].mean().pow(0.5)
    ax.bar(dow.index, dow.values, color="steelblue", edgecolor="black")
    ax.set_title("RMSLE по дням недели")
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"])
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "05_error_by_dow.png", dpi=100)
    plt.close()
    print("  Сохранён: 05_error_by_dow.png")

    # 2. По промо
    fig, ax = plt.subplots()
    promo = val.groupby("onpromotion")["sq_error"].mean().pow(0.5)
    ax.bar(["Без промо", "Промо"], promo.values, color=["gray", "red"], edgecolor="black")
    ax.set_title("RMSLE: промо vs без промо")
    for i, v in enumerate(promo.values):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "06_error_by_promo.png", dpi=100)
    plt.close()
    print("  Сохранён: 06_error_by_promo.png")

    # 3. Остатки
    fig, ax = plt.subplots()
    ax.hist(val["error"], bins=80, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title("Распределение остатков (pred - true)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "07_residuals.png", dpi=100)
    plt.close()
    print("  Сохранён: 07_residuals.png")

    # 4. По магазинам
    fig, ax = plt.subplots(figsize=(12, 6))
    store_err = val.groupby("store_nbr")["sq_error"].mean().pow(0.5).sort_values()
    ax.barh(range(len(store_err)), store_err.values, color="coral", edgecolor="black")
    ax.set_yticks(range(len(store_err)))
    ax.set_yticklabels(store_err.index)
    ax.set_title("RMSLE по магазинам")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "08_error_by_store.png", dpi=100)
    plt.close()
    print("  Сохранён: 08_error_by_store.png")

    print("\nТоп-5 магазинов с наибольшей ошибкой:")
    print(store_err.tail(5))
    print("\nТоп-5 магазинов с наименьшей ошибкой:")
    print(store_err.head(5))

    metrics = {
        "overall_rmsle": rmsle(y_val, pred),
        "rmsle_no_promo": promo.iloc[0],
        "rmsle_promo": promo.iloc[1],
        "mean_error": val["error"].mean(),
        "std_error": val["error"].std(),
    }
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "error_analysis_metrics.csv", index=False)
    print(f"\nМетрики: {RESULTS_DIR / 'error_analysis_metrics.csv'}")


if __name__ == "__main__":
    main()