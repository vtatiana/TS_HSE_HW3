"""Разведочный анализ данных (EDA)."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_DIR, RESULTS_DIR

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 11


def load_data():
    print("Загружаем train_subsample.parquet...")
    train = pd.read_parquet(DATA_DIR / "train_subsample.parquet")
    print(f"  train: {train.shape}")
    return train


def basic_stats(train):
    print("\n" + "=" * 60)
    print("БАЗОВАЯ СТАТИСТИКА")
    print("=" * 60)
    print(f"Период: {train['date'].min()} -> {train['date'].max()}")
    print(f"Магазинов: {train['store_nbr'].nunique()}")
    print(f"Товаров: {train['item_nbr'].nunique()}")
    print(f"Средние: {train['unit_sales'].mean():.3f}")
    print(f"Медиана: {train['unit_sales'].median():.3f}")
    print(f"Доля нулевых: {(train['unit_sales'] == 0).mean() * 100:.2f}%")
    print(f"Доля промо: {train['onpromotion'].mean() * 100:.2f}%")


def plot_sales_distribution(train):
    print("\nСтроим распределение...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(train["unit_sales"].clip(upper=50), bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_title("Распределение продаж (обрезано на 50)")
    axes[1].hist(np.log1p(train["unit_sales"]), bins=50, edgecolor="black", alpha=0.7, color="orange")
    axes[1].set_title("Распределение log1p(unit_sales)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "01_sales_distribution.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("  Сохранён: 01_sales_distribution.png")


def plot_seasonality(train):
    print("\nСтроим сезонность...")
    df = train.copy()
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    dow = df.groupby("dayofweek")["unit_sales"].mean()
    axes[0, 0].bar(dow.index, dow.values, color="steelblue", edgecolor="black")
    axes[0, 0].set_title("Средние продажи по дням недели")
    axes[0, 0].set_xticks(range(7))
    axes[0, 0].set_xticklabels(["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"])

    month = df.groupby("month")["unit_sales"].mean()
    axes[0, 1].bar(month.index, month.values, color="coral", edgecolor="black")
    axes[0, 1].set_title("Средние продажи по месяцам")

    daily = df.groupby("date")["unit_sales"].sum()
    axes[1, 0].plot(daily.index, daily.values, linewidth=0.5, color="green")
    axes[1, 0].set_title("Общий тренд продаж по дням")

    year = df.groupby("year")["unit_sales"].mean()
    axes[1, 1].bar(year.index, year.values, color="purple", edgecolor="black")
    axes[1, 1].set_title("Средние продажи по годам")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "02_seasonality.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("  Сохранён: 02_seasonality.png")


def plot_promo_effect(train):
    print("\nСтроим эффект промо...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    promo_stats = train.groupby("onpromotion")["unit_sales"].agg(["mean", "median", "count"])
    print("\nПродажи по промо:")
    print(promo_stats)
    axes[0].bar(["Без промо", "Промо"], promo_stats["mean"].values, color=["gray", "red"], edgecolor="black")
    axes[0].set_title("Средние продажи: без промо vs промо")
    for i, v in enumerate(promo_stats["mean"].values):
        axes[0].text(i, v + 0.1, f"{v:.2f}", ha="center")

    df = train.copy()
    df["dayofweek"] = df["date"].dt.dayofweek
    promo_by_dow = df.groupby("dayofweek")["onpromotion"].mean() * 100
    axes[1].bar(promo_by_dow.index, promo_by_dow.values, color="orange", edgecolor="black")
    axes[1].set_title("Доля промо по дням недели")
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"])

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "03_promo_effect.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("  Сохранён: 03_promo_effect.png")


def top_stores_items(train):
    print("\nТоп магазинов и товаров...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    top_stores = train.groupby("store_nbr")["unit_sales"].sum().sort_values(ascending=False).head(10)
    axes[0].barh(range(len(top_stores)), top_stores.values, color="steelblue", edgecolor="black")
    axes[0].set_yticks(range(len(top_stores)))
    axes[0].set_yticklabels(top_stores.index)
    axes[0].set_title("Топ-10 магазинов")
    axes[0].invert_yaxis()

    top_items = train.groupby("item_nbr")["unit_sales"].sum().sort_values(ascending=False).head(10)
    axes[1].barh(range(len(top_items)), top_items.values, color="coral", edgecolor="black")
    axes[1].set_yticks(range(len(top_items)))
    axes[1].set_yticklabels(top_items.index)
    axes[1].set_title("Топ-10 товаров")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "04_top_stores_items.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("  Сохранён: 04_top_stores_items.png")


def main():
    print("=" * 60)
    print("EDA")
    print("=" * 60)
    train = load_data()
    basic_stats(train)
    plot_sales_distribution(train)
    plot_seasonality(train)
    plot_promo_effect(train)
    top_stores_items(train)
    print("\n" + "=" * 60)
    print("EDA ЗАВЕРШЁН")
    print("=" * 60)


if __name__ == "__main__":
    main()