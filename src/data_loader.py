"""загрузка и препроцессинг данных"""
import pandas as pd
import numpy as np
from pathlib import Path
from config import (
    DATA_DIR,DATE_COL,TARGET,STORE_COL,ITEM_COL,
    N_STORES,N_ITEMS,RANDOM_STATE
)


def get_selected_stores_items():
    """отбираем магазины и товары для сабсемпла."""
    st= pd.read_csv(DATA_DIR/"stores.csv")
    sel_st= st["store_nbr"].head(N_STORES).tolist()
    #  идём по train чанками, чтобы не грузить все 5ГБ в память
    print("вычислить частоты товаров...")
    cnt= {}
    for ch in pd.read_csv(
        DATA_DIR/"train.csv",
        usecols=[ITEM_COL],
        chunksize=5_000_000,
    ):
        c= ch[ITEM_COL].value_counts()
        for it,n in c.items():
            cnt[it]= cnt.get(it,0) +n

    #  топ-N по частоте
    top= sorted(cnt.items(),key=lambda x: -x[1])[:N_ITEMS]
    sel_it= [it for it,_ in top]
    print(f"выбрали {len(sel_st)} магазинов и {len(sel_it)} товаров")
    return   sel_st,sel_it


def load_train_subsample(sel_st,sel_it):
    """читаем train чанками, оставляем в итоге только нужные пары."""
    print("загружаем train по чанкам...")
    chunks= []
    for ch in pd.read_csv(
        DATA_DIR/"train.csv",
        dtype= {
            STORE_COL: "int8",
            ITEM_COL: "int32",
            "unit_sales": "float32",
                 },
        parse_dates=[DATE_COL],
        chunksize=5_000_000,
    ):
        m= ch[STORE_COL].isin(sel_st) & ch[ITEM_COL].isin(sel_it)
        f= ch[m]
        if len(f) > 0:
            chunks.append(f)
        print(f"  чанков: {len(chunks)},строк: {sum(len(c) for c in chunks)}")

    tr= pd.concat(chunks,ignore_index=True)
    print(f"тrain загружен: {tr.shape}")
    return   tr


def load_auxiliary():
    """загружаем вспомогательные таблицы."""
    st= pd.read_csv(DATA_DIR/"stores.csv")
    it= pd.read_csv(DATA_DIR/"items.csv")
    oil= pd.read_csv(DATA_DIR/"oil.csv",parse_dates=[DATE_COL])
    hol= pd.read_csv(DATA_DIR/"holidays_events.csv",parse_dates=[DATE_COL])
    trx= pd.read_csv(DATA_DIR/"transactions.csv",parse_dates=[DATE_COL])
    return   st,it,oil,hol,trx


if __name__ == "__main__":
    sel_st,sel_it= get_selected_stores_items()
    tr= load_train_subsample(sel_st,sel_it)
    st,it,oil,hol,trx= load_auxiliary()
    print("\n итоги :")
    print(f"train: {tr.shape}")
    print(f"stores: {st.shape}")
    print(f"items: {it.shape}")
    print(f"oil: {oil.shape}")
    print(f"holidays: {hol.shape}")
    print(f"transactions: {trx.shape}")
    print(tr.head())