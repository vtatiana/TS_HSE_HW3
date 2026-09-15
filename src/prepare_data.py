"""готовим подвыборку train и сохраняем в parquet для компакости."""
import pandas as pd
from config import DATA_DIR
from data_loader import (
    get_selected_stores_items,
    load_train_subsample,
)


def clean(train):
    print("\nчистим данные...")
    train["onpromotion"]= train["onpromotion"].fillna(False).astype(bool).astype("int8")
    train["unit_sales"]= train["unit_sales"].clip(lower=0).astype("float32")
    train["store_nbr"]= train["store_nbr"].astype("int8")
    train["item_nbr"]= train["item_nbr"].astype("int32")
    train["id"]= train["id"].astype("int64")
    print(f"после очистки: {train.shape}")
    print(f"onpromotion: {train['onpromotion'].unique()}")
    print(f"unit_sales: min={train['unit_sales'].min()},max={train['unit_sales'].max()},mean={train['unit_sales'].mean():.3f}")
    return   train


def main():
    
    print("подготовка данных")
    
    sel_st,sel_it= get_selected_stores_items()
    tr= load_train_subsample(sel_st,sel_it)
    tr= clean(tr)

    out= DATA_DIR/"train_subsample.parquet"
    print(f"\nсохраняем в {out}...")
    tr.to_parquet(out,index=False,compression="snappy")

    pd.DataFrame({"store_nbr": sel_st}).to_csv(DATA_DIR/"selected_stores.csv",index=False)
    pd.DataFrame({"item_nbr": sel_it}).to_csv(DATA_DIR/"selected_items.csv",index=False)
    print("списки сохранились,ура")

    print(f"\nсколько строк: {pd.read_parquet(out).shape}")



if __name__ == "__main__":
    main()