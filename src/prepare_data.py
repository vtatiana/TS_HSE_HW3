"""Готовим подвыборку train и сохраняем в parquet."""
import pandas as pd
from config import DATA_DIR
from data_loader import (
    get_selected_stores_items,
    load_train_subsample,
)


def clean_train(train):
    print("\nЧистим данные...")
    train["onpromotion"] = train["onpromotion"].fillna(False).astype(bool).astype("int8")
    train["unit_sales"] = train["unit_sales"].clip(lower=0).astype("float32")
    train["store_nbr"] = train["store_nbr"].astype("int8")
    train["item_nbr"] = train["item_nbr"].astype("int32")
    train["id"] = train["id"].astype("int64")
    print(f"После очистки: {train.shape}")
    print(f"onpromotion: {train['onpromotion'].unique()}")
    print(f"unit_sales: min={train['unit_sales'].min()}, max={train['unit_sales'].max()}, mean={train['unit_sales'].mean():.3f}")
    return train


def main():
    print("=" * 60)
    print("ПОДГОТОВКА ДАННЫХ")
    print("=" * 60)

    selected_stores, selected_items = get_selected_stores_items()
    train = load_train_subsample(selected_stores, selected_items)
    train = clean_train(train)

    out_path = DATA_DIR / "train_subsample.parquet"
    print(f"\nСохраняем в {out_path}...")
    train.to_parquet(out_path, index=False, compression="snappy")
    print(f"Готово! {out_path.stat().st_size / 1024**2:.1f} МБ")

    pd.DataFrame({"store_nbr": selected_stores}).to_csv(DATA_DIR / "selected_stores.csv", index=False)
    pd.DataFrame({"item_nbr": selected_items}).to_csv(DATA_DIR / "selected_items.csv", index=False)
    print("Списки магазинов и товаров сохранены.")

    print(f"\nПроверка: {pd.read_parquet(out_path).shape}")
    print("=" * 60)
    print("ЗАВЕРШЕНО")
    print("=" * 60)


if __name__ == "__main__":
    main()