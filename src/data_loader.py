"""Загрузка и предобработка данных Favorita."""
import pandas as pd
import numpy as np
from pathlib import Path
from config import (
    DATA_DIR, DATE_COL, TARGET, STORE_COL, ITEM_COL,
    N_STORES, N_ITEMS, RANDOM_STATE
)


def get_selected_stores_items():
    """Определяем, какие магазины и товары берём в подвыборку."""
    stores = pd.read_csv(DATA_DIR / "stores.csv")
    selected_stores = stores["store_nbr"].head(N_STORES).tolist()

    print("Считаем частоты товаров (по чанкам)...")
    item_counts = {}
    for chunk in pd.read_csv(
        DATA_DIR / "train.csv",
        usecols=[ITEM_COL],
        chunksize=5_000_000,
    ):
        counts = chunk[ITEM_COL].value_counts()
        for item, cnt in counts.items():
            item_counts[item] = item_counts.get(item, 0) + cnt

    top_items = sorted(item_counts.items(), key=lambda x: -x[1])[:N_ITEMS]
    selected_items = [item for item, _ in top_items]

    print(f"Выбрано {len(selected_stores)} магазинов и {len(selected_items)} товаров")
    return selected_stores, selected_items


def load_train_subsample(selected_stores, selected_items):
    """Читаем train чанками, оставляем только нужные магазины/товары."""
    print("Загружаем train (по чанкам)...")
    chunks = []
    for chunk in pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={
            STORE_COL: "int8",
            ITEM_COL: "int32",
            "unit_sales": "float32",
        },
        parse_dates=[DATE_COL],
        chunksize=5_000_000,
    ):
        mask = chunk[STORE_COL].isin(selected_stores) & chunk[ITEM_COL].isin(selected_items)
        filtered = chunk[mask]
        if len(filtered) > 0:
            chunks.append(filtered)
        print(f"  Чанков обработано: {len(chunks)}, строк: {sum(len(c) for c in chunks)}")

    train = pd.concat(chunks, ignore_index=True)
    print(f"Train загружен: {train.shape}")
    return train


def load_auxiliary():
    """Загружаем вспомогательные датасеты."""
    stores = pd.read_csv(DATA_DIR / "stores.csv")
    items = pd.read_csv(DATA_DIR / "items.csv")
    oil = pd.read_csv(DATA_DIR / "oil.csv", parse_dates=[DATE_COL])
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=[DATE_COL])
    transactions = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=[DATE_COL])
    return stores, items, oil, holidays, transactions


if __name__ == "__main__":
    selected_stores, selected_items = get_selected_stores_items()
    train = load_train_subsample(selected_stores, selected_items)
    stores, items, oil, holidays, transactions = load_auxiliary()

    print("\n=== Итог ===")
    print(f"train: {train.shape}")
    print(f"stores: {stores.shape}")
    print(f"items: {items.shape}")
    print(f"oil: {oil.shape}")
    print(f"holidays: {holidays.shape}")
    print(f"transactions: {transactions.shape}")
    print("\nПервые строки train:")
    print(train.head())