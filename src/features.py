"""Генерация признаков для ML-моделей."""
import pandas as pd
import numpy as np
from config import DATA_DIR

TARGET= "unit_sales"


def load_merge():
    print("загружаем данные")
    tr= pd.read_parquet(DATA_DIR/"train_subsample.parquet")
    st= pd.read_csv(DATA_DIR/"stores.csv")
    it= pd.read_csv(DATA_DIR/"items.csv")
    oil= pd.read_csv(DATA_DIR/"oil.csv",parse_dates=["date"])
    hol= pd.read_csv(DATA_DIR/"holidays_events.csv",parse_dates=["date"])
    trx= pd.read_csv(DATA_DIR/"transactions.csv",parse_dates=["date"])
    print(f"  train: {tr.shape}")
    df= tr.merge(st,on="store_nbr",how="left")
    df= df.merge(it,on="item_nbr",how="left")
    oil= oil.sort_values("date").rename(columns={"dcoilwtico": "oil"})
    oil["oil"]= oil["oil"].ffill().bfill()
    
    df= df.merge(oil,on="date",how="left")

    nat= hol[hol["locale"]== "National"]["date"].unique()
    df["is_holiday"]= df["date"].isin(nat).astype("int8")

    df= df.merge(trx,on=["date","store_nbr"],how="left")
    df["transactions"]= df["transactions"].fillna(0).astype("float32")

    print(f"  после merge: {df.shape}")
    return   df


def add_time(df):
    print("временные фичи")
    d= df["date"].dt
    df["dayofweek"]= d.dayofweek.astype("int8")
    df["month"]= d.month.astype("int8")
    df["day"]= d.day.astype("int8")
    df["year"]= d.year.astype("int16")
    df["weekofyear"]= d.isocalendar().week.astype("int8")
    df["dayofyear"]= d.dayofyear.astype("int16")
    df["dow_sin"]= np.sin(2*np.pi*df["dayofweek"]/7).astype("float32")
    df["dow_cos"]= np.cos(2*np.pi*df["dayofweek"]/7).astype("float32")
    df["month_sin"]= np.sin(2*np.pi*df["month"]/12).astype("float32")
    df["month_cos"]= np.cos(2*np.pi*df["month"]/12).astype("float32")
    return   df


def add_lag(df):
    print("лаговые фичи")
    df= df.sort_values(["store_nbr","item_nbr","date"]).reset_index(drop=True)
    g= df.groupby(["store_nbr","item_nbr"])[TARGET]
    for lg in [1,7,14,28]:
        df[f"lag_{lg}"]= g.shift(lg).astype("float32")
    return   df


def add_roll(df):
    print("скользящие фичи")
    g= df.groupby(["store_nbr","item_nbr"])[TARGET]
    for w in [7,30,90]:
        sh= g.shift(1)
        df[f"roll_mean_{w}"]= sh.rolling(w,min_periods=1).mean().astype("float32")
        df[f"roll_std_{w}"]= sh.rolling(w,min_periods=1).std().astype("float32")
    return   df


def add_promo(df):
    print("промо фичи")
    df["promo_count_store"]= df.groupby(["date","store_nbr"])["onpromotion"].transform("sum").astype("int16")
    df["promo_ratio_store"]= (df["promo_count_store"]/df.groupby(["date","store_nbr"])["onpromotion"].transform("count")).astype("float32")
    return   df


def main():
    print("feature engineering")
    df= load_merge()
    df= add_time(df)
    df= add_lag(df)
    df= add_roll(df)
    df= add_promo(df)
    df["target_log"]= np.log1p(df[TARGET].clip(lower=0)).astype("float32")
    before= len(df)
    df= df.dropna(subset=["lag_28"]).reset_index(drop=True)
    print(f"\nудалили строк совсем без лагов: {before -len(df)}")
    cat_cols= ["family","city","state","type"]
    int_cols= ["store_nbr","item_nbr","cluster","class","perishable"]
    for col in cat_cols:
        if col in df.columns:
            df[col]= df[col].astype("category")
    for col in int_cols:
        if col in df.columns:
            df[col]= df[col].astype("int32")

    print(f"\nитоги: {df.shape}")
    out= DATA_DIR/"features.parquet"
    print(f"сохраняем {out}")
    df.to_parquet(out,index=False,compression="snappy")
    print(f"сделалось->{out.stat().st_size/1024**2:.1f} ")


if __name__ == "__main__":
    main()